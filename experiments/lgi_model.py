from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch_geometric.data import HeteroData

from experiments.protein_encoding import ENCODER_MAP, EMBED_SIZES
from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.model.downstream import DownstreamGGIN
from gifflar.model.glylm import GlycanLM
from gifflar.model.utils import EmbeddingStorage, GIFFLARPooling, LectinStorage, get_prediction_head
from gifflar.utils import get_metrics

THRESHOLD = 0.5

GLYCAN_ENCODERS = {
    "gifflar": DownstreamGGIN,
    "sweetnet": SweetNetLightning,
    "glylm": GlycanLM,
}

class LGI_Model(LightningModule):
    def __init__(
            self,
            glycan_encoder_name: str,
            glycan_encoder_args: dict[str, Any],
            lectin_encoder: str,
            le_layer_num: int,
            add_tasks: list[str, str] = [],
            **kwargs: Any,
    ):
        """
        Initialize the LGI model, a model for predicting lectin-glycan interactions.

        Args:
            glycan_encoder: The glycan encoder model
            lectin_encoder: The lectin encoder model
            le_layer_num: The number of layers to use in the lectin encoder
            kwargs: Additional arguments
        """
        super().__init__()

        self.glycan_encoder = GLYCAN_ENCODERS[glycan_encoder_name](**glycan_encoder_args)
        if glycan_encoder_name == "glylm":
            self.glycan_embeddings = EmbeddingStorage(
                encoder=self.glycan_encoder.encoder, # get the pipeline out of the model (glylm.py:35)
                name="glylm_" + Path(glycan_encoder_args["model_dir"]).parent.name,
                path=kwargs["root_dir"],
            )
        
        self.lectin_encoder = lectin_encoder
        self.le_layer_num = le_layer_num
        self.add_tasks = add_tasks

        self.lectin_embeddings = LectinStorage(
            encoder=ENCODER_MAP[lectin_encoder](le_layer_num), 
            lectin_encoder=lectin_encoder, 
            le_layer_num=le_layer_num, 
            path=kwargs["root_dir"],
        )
        
        self.combined_dim = self.glycan_encoder.hidden_dim + EMBED_SIZES[lectin_encoder]

        self.head, self.loss, self.metrics = get_prediction_head(self.combined_dim, 1, "regression", size="large")
        self.add_metrics = [get_metrics(task=task, n_outputs=1, prefix=name) for name, task in self.add_tasks]

    def to(self, device: torch.device):
        super(LGI_Model, self).to(device)
        self.glycan_encoder.to(device)
        self.head.to(device)
        if hasattr(self, "glycan_embeddings"):
            self.glycan_embeddings.to(device)
        self.lectin_embeddings.to(device)

        self.metrics = {split: metric.to(device) for split, metric in self.metrics.items()}
        self.add_metrics = [{split: metric.to(device) for split, metric in metrics.items()} for metrics in self.add_metrics]
        
        return self

    def forward(self, data: HeteroDataBatch) -> dict[str, torch.Tensor]:
        if hasattr(self, "glycan_embeddings"):
            glycan_embed = self.glycan_embeddings.batch_query(data["IUPAC"])
        else:
            glycan_embed = self.glycan_encoder(data)["graph_embed"]
        lectin_embed = self.lectin_embeddings.batch_query(data["aa_seq"])
        combined = torch.cat([glycan_embed, lectin_embed], dim=-1)
        pred = self.head(combined)

        return {
            # "glycan_node_embeds": glycan_embed["node_embed"],
            "glycan_graph_embeds": glycan_embed,  # ["graph_embed"],
            "lectin_embeds": lectin_embed,
            "preds": pred,
        }

    def shared_step(self, batch: HeteroData, stage: str, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """
        Compute the shared step of the model.

        Args:
            data: The data to process
            stage: The stage of the model

        Returns:
            A dictionary containing the loss and the metrics
        """
        fwd_dict = self(batch)
        fwd_dict["labels"] = batch["y"]
        fwd_dict["preds"] = fwd_dict["preds"].reshape(-1)
        if dataloader_idx == 0:
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"])
            self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]), add_dataloader_idx=False)
        else:
            name, task = self.add_tasks[dataloader_idx - 1]
            if task == "classification":
                fwd_dict["loss"] = nn.BCEWithLogitsLoss()(fwd_dict["preds"], fwd_dict["labels"].float())
                self.add_metrics[dataloader_idx - 1][stage].update(fwd_dict["preds"], fwd_dict["labels"])
                self.log(f"{stage}/{name}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]), add_dataloader_idx=False)
            elif task in {"regression", "spectrum"}:
                pass
            else:
                raise ValueError(f"Task {task} is not supported")
        return fwd_dict

    def training_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the training step of the model"""
        return self.shared_step(batch, "train", batch_idx, dataloader_idx)

    def validation_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the validation step of the model"""
        return self.shared_step(batch, "val", batch_idx, dataloader_idx)

    def test_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the testing step of the model"""
        return self.shared_step(batch, "test", batch_idx, dataloader_idx)

    def predict_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        fwd_dict = self(batch, "test", batch_idx, dataloader_idx)
        fwd_dict["IUPAC"] = batch["IUPAC"]
        fwd_dict["seq"] = batch["aa_seq"]
        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]):
        """
        Compute the shared end of the model.

        Args:
            stage: The stage of the model
        """
        for loggable in [self.metrics[stage]] + [self.add_metrics[i][stage] for i in range(len(self.add_metrics))]:
            if list(loggable.values())[0].update_count == 0:
                continue
            metrics = loggable.compute()
            self.log_dict(metrics)
            loggable.reset()

    def on_train_epoch_end(self) -> None:
        """Compute the end of the training epoch"""
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        """Compute the end of the validation"""
        self.shared_end("val")
        self.lectin_embeddings.close()

    def on_test_epoch_end(self) -> None:
        """Compute the end of the testing"""
        self.shared_end("test")

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler of the model"""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5),
            "monitor": "val/loss",
        }
