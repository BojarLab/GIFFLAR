from typing import Any, Literal

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import HeteroData

from experiments.lgi_model import LGI_Model
from experiments.protein_encoding import EMBED_SIZES
from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.baselines.sweetnet import SweetNetLightning
from gifflar.utils import get_metrics


def sigmoid_cosine_distance_p(x, y, p=1):
    sig = torch.nn.Sigmoid()
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - sig(cosine_sim(x, y))) ** p


def cosine_distance_p(x, y, p=1):
    cosine_sim = torch.nn.CosineSimilarity()
    return (1 - cosine_sim(x, y)) ** p


DISTANCES = {
    "sigmoid": sigmoid_cosine_distance_p,
    "cosine": cosine_distance_p,
}


class ContrastLGIModel(LGI_Model):
    def __init__(
            self,
            glycan_encoder: GlycanGIN | SweetNetLightning,
            lectin_encoder: str,
            le_layer_num: int,
            latent_dim: int = 256,
            margin_distance: Literal["sigmoid", "cosine"] = "cosine",
            margin: float = 0.25,
            add_tasks: list[str, str] = [],
            ** kwargs: Any,
    ):
        super(ContrastLGIModel, self).__init__(glycan_encoder, lectin_encoder, le_layer_num, **kwargs)

        self.latent_dim = latent_dim
        self.add_tasks = add_tasks

        self.glycan_red = nn.Linear(glycan_encoder.hidden_dim, latent_dim)
        self.lectin_red = nn.Linear(EMBED_SIZES[lectin_encoder], latent_dim)

        self.pred_loss = nn.MSELoss()
        self.embed_loss = nn.TripletMarginWithDistanceLoss(distance_function=DISTANCES[margin_distance], margin=margin)
        self.add_metrics = [get_metrics(task=task, n_outputs=1, prefix=name) for name, task in self.add_tasks]

    def to(self, device: torch.device):
        super(ContrastLGIModel, self).to(device)
        self.glycan_red.to(device)
        self.lectin_red.to(device)
        return self

    def forward(self, data: HeteroDataBatch, decoys: HeteroDataBatch | None) -> dict[str, torch.Tensor]:
        glycan_embed = self.glycan_encoder(data)["graph_embed"]
        glycan_small = self.glycan_red(glycan_embed)

        lectin_embed = self.lectin_embeddings.batch_query(data["aa_seq"])
        lectin_small = self.lectin_red(lectin_embed)

        fwd_dict = {
            "glycan": glycan_small,
            "lectin": lectin_small,
        }
        if decoys is not None:
            decoy_embed = self.glycan_encoder(decoys)["graph_embed"]
            decoy_small = self.glycan_red(decoy_embed)
            fwd_dict["decoy"] = decoy_small

        return fwd_dict

    def shared_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None] | HeteroDataBatch, stage: str, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        if isinstance(batch, tuple):
            batch, decoy = batch
        else:
            decoy = None
        fwd_dict = self.forward(batch, decoy)
        fwd_dict["labels"] = batch["y"].float()

        inter_pred = sigmoid_cosine_distance_p(fwd_dict["glycan"], fwd_dict["lectin"])
        fwd_dict["preds"] = inter_pred
        
        if dataloader_idx == 0:
            inter_loss = self.pred_loss(inter_pred, fwd_dict["labels"])
            fwd_dict["inter_loss"] = inter_loss.float()

            if decoy is not None:
                embed_loss = self.embed_loss(fwd_dict["glycan"], fwd_dict["lectin"], fwd_dict["decoy"])
                fwd_dict["embed_loss"] = embed_loss
                loss = inter_loss + embed_loss # weighting factor?
            else:
                loss = inter_loss
            fwd_dict["loss"] = loss.float()

            self.metrics[stage].update(inter_pred, fwd_dict["labels"])
            self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]), add_dataloader_idx=False)
        else:
            name, task = self.add_tasks[dataloader_idx - 1]
            if task == "classification":
                # fwd_dict["preds"] = (fwd_dict["preds"] > THRESHOLD).float()
                fwd_dict["loss"] = nn.BCEWithLogitsLoss()(fwd_dict["preds"], fwd_dict["labels"].float())
                self.add_metrics[dataloader_idx - 1][stage].update(fwd_dict["preds"], fwd_dict["labels"])
                self.log(f"{stage}/{name}/loss", fwd_dict["loss"], batch_size=len(fwd_dict["preds"]), add_dataloader_idx=False)
            elif task in {"regression", "spectrum"}:
                pass
            else:
                raise ValueError(f"Task {task} is not supported")
        return fwd_dict

    def predict_step(self, batch: tuple[HeteroDataBatch, HeteroDataBatch | None] | HeteroDataBatch, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        fwd_dict = self.forward(batch)
        if isinstance(batch, tuple):
            batch = batch[0]
        fwd_dict["IUPAC"] = batch["IUPAC"]
        fwd_dict["seq"] = batch["aa_seq"]
        return fwd_dict
