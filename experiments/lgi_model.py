from pathlib import Path
from typing import Any, Literal
import copy

import torch
from pytorch_lightning import LightningModule
from torch_geometric.data import HeteroData

from experiments.protein_encoding import ENCODER_MAP, EMBED_SIZES
from gifflar.data.utils import GlycanStorage
from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.model.utils import GIFFLARPooling, get_prediction_head


class LectinStorage(GlycanStorage):
    def __init__(self, lectin_encoder: str, le_layer_num: int, path: str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a lectin_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        self.path = Path(path or "data") / f"{lectin_encoder}_{le_layer_num}.pkl"
        # print("Path:", self.path.resolve())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.encoder = ENCODER_MAP[lectin_encoder](le_layer_num)
        self.data = self._load()

    def query(self, aa_seq: str) -> torch.Tensor:
        if aa_seq not in self.data:
            try:
                self.data[aa_seq] = self.encoder(aa_seq)
            except Exception as e:
                print(e)
                self.data[aa_seq] = None

        return self.data[aa_seq]

    def batch_query(self, aa_seqs) -> torch.Tensor:
        # print([self.query(aa_seq) for aa_seq in aa_seqs])
        results = [self.query(aa_seq) for aa_seq in aa_seqs]
        dummy = None
        for x in results:
            if x is not None:
                dummy = torch.zeros_like(x)
                break
        return torch.stack([dummy if res is None else res for res in results])


class LGI_Model(LightningModule):
    def __init__(
            self,
            glycan_encoder: GlycanGIN | SweetNetLightning,
            lectin_encoder: str,
            le_layer_num: int,
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

        self.glycan_encoder = glycan_encoder
        self.glycan_pooling = GIFFLARPooling("global_mean")
        self.lectin_encoder = lectin_encoder
        self.le_layer_num = le_layer_num

        self.lectin_embeddings = LectinStorage(lectin_encoder, le_layer_num, kwargs["root_dir"])
        self.combined_dim = glycan_encoder.hidden_dim + EMBED_SIZES[lectin_encoder]

        self.head, self.loss, self.metrics = get_prediction_head(self.combined_dim, 1, "regression", size="large")

    def to(self, device: torch.device):
        super(LGI_Model, self).to(device)
        self.glycan_encoder.to(device)
        self.glycan_pooling.to(device)
        self.head.to(device)
        for split, metric in self.metrics.items():
            self.metrics[split] = metric.to(device)

    def forward(self, data: HeteroDataBatch) -> dict[str, torch.Tensor]:
        glycan_embed = self.glycan_encoder(data)
        lectin_embed = self.lectin_embeddings.batch_query(data["aa_seq"])
        combined = torch.cat([glycan_embed["graph_embed"], lectin_embed], dim=-1)
        pred = self.head(combined)

        return {
            "glycan_node_embeds": glycan_embed["node_embed"],
            "glycan_graph_embeds": glycan_embed["graph_embed"],
            "lectin_embeds": lectin_embed,
            "preds": pred,
        }

    def shared_step(self, batch: HeteroData, stage: str) -> dict[str, torch.Tensor]:
        """
        Compute the shared step of the model.

        Args:
            data: The data to process
            stage: The stage of the model

        Returns:
            A dictionary containing the loss and the metrics
        """
        fwd_dict = self(batch)
        fwd_dict["labels"] = batch["y"]# .reshape(-1)
        fwd_dict["preds"] = fwd_dict["preds"].reshape(-1)
        fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])
        self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"])
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]))

        return fwd_dict

    def training_step(self, batch: HeteroData, batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the training step of the model"""
        return self.shared_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the validation step of the model"""
        return self.shared_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int) -> dict[str, torch.Tensor]:
        """Compute the testing step of the model"""
        return self.shared_step(batch, "test")

    def predict_step(self, batch: HeteroData, batch_idx: int) -> dict[str, torch.Tensor]:
        fwd_dict = self(batch)
        fwd_dict["IUPAC"] = batch["IUPAC"]
        fwd_dict["seq"] = batch["aa_seq"]
        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]):
        """
        Compute the shared end of the model.

        Args:
            stage: The stage of the model
        """
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()

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
