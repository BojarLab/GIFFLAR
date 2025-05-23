from typing import Literal, Optional, Any

import torch
import matplotlib
matplotlib.use('TkAgg',force=True)
from glycowork.glycan_data.loader import lib
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.utils import MultiEmbedding, get_gin_layer
from gifflar.pretransforms import LaplacianPE, RandomWalkPE
from gifflar.utils import bond_map, atom_map

PRE_TRANSFORMS = {
    "LaplacianPE": LaplacianPE,
    "RandomWalkPE": RandomWalkPE,
}


class GlycanGIN(LightningModule):
    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, batch_size: int = 32,
                 pre_transform_args: Optional[dict] = None, **kwargs: Any):
        """
        Initialize the GlycanGIN model, the base for all DL-models in this package

        Args:
            feat_dim: The feature dimension of the model
            hidden_dim: The hidden dimension of the model
            num_layers: The number of GIN layers to use
            pre_transform_args: A dictionary of pre-transforms to apply to the input data
            kwargs: Additional arguments (ignored)
        """
        super().__init__()

        # Preparation for additional information in the embeddings of the nodes
        rand_dim = feat_dim
        self.addendum = []
        if pre_transform_args is not None:
            for name, args in pre_transform_args.items():
                if name in PRE_TRANSFORMS:
                    self.addendum.append(PRE_TRANSFORMS[name].attr_name)
                    rand_dim -= args["dim"]

        # Set up the learnable embeddings for all node types
        self.embedding = MultiEmbedding({
            "atoms": nn.Embedding(len(atom_map) + 2, rand_dim, _freeze=True),
            "bonds": nn.Embedding(len(bond_map) + 2, rand_dim, _freeze=True),
            "monosacchs": nn.Embedding(2368, rand_dim, _freeze=True),  # len(lib) + 2
        })

        # Define the GIN layers to embed messages between nodes
        self.convs = torch.nn.ModuleList()
        dims = [feat_dim]
        if feat_dim <= hidden_dim // 2:
            dims += [hidden_dim // 2]
        else:
            dims += [hidden_dim]
        dims += [hidden_dim] * (num_layers - 1)
        for i in range(num_layers):
            self.convs.append(HeteroConv({
                key: get_gin_layer(dims[i], dims[i + 1]) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Compute the node embeddings.

        Args:
            batch: The batch of data to process

        Returns:
            node_embed: The node embeddings
        """
        for key in batch.x_dict.keys():
            # Compute random encodings for the atom type and include positional encodings
            pes = [self.embedding.forward(batch.x_dict[key], key)]
            for pe in self.addendum:
                pes.append(batch[f"{key}_{pe}"])

            batch.x_dict[key] = torch.concat(pes, dim=1)

        for conv in self.convs:
            if isinstance(conv, HeteroConv):
                batch.x_dict = conv(batch.x_dict, batch.edge_index_dict)
            else:  # the layer is an activation function from the RGCN
                batch.x_dict = conv(batch.x_dict)

        return batch.x_dict

    def shared_step(self, batch: HeteroData, stage: Literal["train", "val", "test"], batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def training_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the training step of the model"""
        return self.shared_step(batch, "train")

    def validation_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the validation step of the model"""
        return self.shared_step(batch, "val")

    def test_step(self, batch: HeteroData, batch_idx: int = 0, dataloader_idx: int = 0) -> dict[str, torch.Tensor]:
        """Compute the testing step of the model"""
        return self.shared_step(batch, "test")

    def shared_end(self, stage: Literal["train", "val", "test"]) -> dict[str, torch.Tensor]:
        raise NotImplementedError()

    def on_train_epoch_end(self) -> None:
        """Compute the end of the training epoch"""
        self.shared_end("train")

    def on_validation_epoch_end(self) -> None:
        """Compute the end of the validation"""
        self.shared_end("val")

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
