from typing import Literal, Optional, Any

import torch
from torch import nn
from torch_geometric.data import HeteroData

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.base import GlycanGIN
from gifflar.model.utils import GIFFLARPooling, get_prediction_head, get_spectrum_prediction_head
from gifflar.utils import get_metrics


class DownstreamGGIN(GlycanGIN):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            output_dim: int = 1,
            task: Literal["regression", "classification", "multilabel", "spectrum", "spectrum"] | None = None,
            num_layers: int = 3,
            batch_size: int = 32,
            pre_transform_args: Optional[dict] = None,
            **kwargs: Any,
    ):
        """
        Initialize the DownstreamGGIN model, a downstream model from a pre-trained GlycanGIN model.

        Args:
            hidden_dim: The hidden dimension of the model
            output_dim: The output dimension of the model
            task: The task to perform, either "regression", "classification", "multilabel", "spectrum"
            num_layers: The number of GIN layers to use
            batch_size: The batch size to use
            pre_transform_args: A dictionary of pre-transforms to apply to the input data
            kwargs: Additional arguments
        """
        super().__init__(feat_dim, hidden_dim, num_layers, batch_size, pre_transform_args)

        self.pooling = GIFFLARPooling(kwargs.get("pooling", "global_mean"))
        self.output_dim = output_dim
        self.task = task
        if self.task is not None:
            self.head, self.loss, self.metrics = get_prediction_head(hidden_dim, self.output_dim, self.task)

    def to(self, device: torch.device) -> "DownstreamGGIN":
        """
        Move the model to the specified device.

        Args:
            device: The device to move the model to

        Returns:
            self: The model moved to the specified device
        """
        super(DownstreamGGIN, self).to(device)
        if isinstance(self.pooling, GIFFLARPooling):
            self.pooling.to(device)
        if self.task is not None:
            for split, metric in self.metrics.items():
                self.metrics[split] = metric.to(device)
        return self

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            batch: The batch of data to process

        Returns:
            A dictionary containing:
                node_embed: The node embeddings
                graph_embed: The graph embeddings
                preds: The predictions of the model
        """
        node_embed = super().forward(batch)
        graph_embed = self.pooling(node_embed, batch.batch_dict)

        pred = None
        if self.task is not None:
            pred = self.head(graph_embed)

        return {
            "node_embed": node_embed,
            "graph_embed": graph_embed,
            "preds": pred,
        }

    def shared_step(self, batch: HeteroData, stage: Literal["train", "val", "test"], batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """
        Shared step for training, validation and testing steps.

        Args:
            batch: The batch of data to process
            stage: The stage of the model, either "train", "val" or "test"

        Returns:
            A dictionary containing:
                node_embed: The node embeddings
                graph_embed: The graph embeddings
                preds: The predictions of the model
                labels: The true labels of the data
                loss: The loss of the model
        """
        fwd_dict = self.forward(batch)
        fwd_dict["labels"] = batch["y_oh"] if self.task == "multilabel" else batch["y"]

        # Adjust the predictions to match the output dimension and eventually apply the sigmoid function
        if self.task != "multilabel":
            if list(fwd_dict["preds"].shape) == [len(batch["y"]), 1]:
                fwd_dict["preds"] = fwd_dict["preds"][:, 0]

        # Compute the loss based on the task and the number of outputs to predict
        if self.output_dim == 1 or self.task == "multilabel":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape).float())
        elif self.task == "classification":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        elif self.task == "spectrum":
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"], target=torch.ones(len(fwd_dict["labels"])).to(fwd_dict["labels"].device))
        else:
            fwd_dict["loss"] = self.loss(fwd_dict["preds"], fwd_dict["labels"])

        # Update the metrics based on the task and the number of outputs to predict
        if self.task == "classification" and self.output_dim > 1:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape[:-1]))
        elif self.task == "spectrum":
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape), precursor_masses=batch["red_mz"])
        else:
            self.metrics[stage].update(fwd_dict["preds"], fwd_dict["labels"].reshape(fwd_dict["preds"].shape))

        # Log the loss of the model
        self.log(f"{stage}/loss", fwd_dict["loss"], batch_size=self.batch_size)
        return fwd_dict

    def shared_end(self, stage: Literal["train", "val", "test"]) -> None:
        """
        Shared end for training, validation and testing steps.

        Args:
            stage: The stage of the model, either "train", "val" or "test"
        """
        metrics = self.metrics[stage].compute()
        self.log_dict(metrics)
        self.metrics[stage].reset()
