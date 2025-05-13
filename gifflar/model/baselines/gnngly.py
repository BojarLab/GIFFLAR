from typing import Literal, Any
from collections import OrderedDict

import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN
from gifflar.model.utils import GIFFLARPooling

"""
# Model properties extracted from GNNGLY
Learning Rate:    0
Dropout:          0.1
Batch Size:     256
Max Epoch:      200
Num GNN layers:   5
Patience:         0.1
in_feats:       133   [max degree is 12?!]
hidden_dim:      14
"""


class GNNGLY(DownstreamGGIN):
    """
    Implementation follows the loose description in [LINK] as good as possible. Where the specification is unclear, we
    fill in what sounds reasonable
    """

    def __init__(self, feat_dim: int, hidden_dim: int, num_layers: int, output_dim: int,
                 task: Literal["classification", "regression", "multilabel", "spectrum"], **kwargs: Any):
        """
        Initialize the model following the papers description.

        Args:
            feat_dim: The dimension of the input features
            hidden_dim: The dimension of the hidden layers
            num_layers: The number of GCN layers
            output_dim: The dimension of the output layer
            task: The task to solve
            kwargs: Additional arguments to pass to the model
        """
        super().__init__(feat_dim, hidden_dim, output_dim, task, **kwargs)

        del self.convs

        self.layers = nn.Sequential(OrderedDict(
            [(f"layer{l + 1}", GCNConv((133 if l == 0 else hidden_dim), hidden_dim)) for l in range(num_layers)]))

        # ASSUMPTION: mean pooling
        del self.pooling
        self.pooling = global_mean_pool  # GIFFLARPooling()

    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Forward the data though the model.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings, the graph embedding, and the final model prediction
        """
        # Extract atomic graph from the heterogeneous graph
        x = batch["gnngly_x"].float()
        batch_ids = batch["gnngly_batch"]
        edge_index = batch["gnngly_edge_index"]

        # Propagate the data through the model
        for layer in self.layers:
            x = layer(x, edge_index)

        # Compute the graph embeddings and make the final prediction based on this
        graph_embed = self.pooling(x, batch_ids)
        pred = self.head(graph_embed)

        return {
            "node_embed": x,
            "graph_embed": graph_embed,
            "preds": pred,
        }
