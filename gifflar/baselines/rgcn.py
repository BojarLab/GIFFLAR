from typing import Literal, Optional, Dict, Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GINConv

from gifflar.model import DownstreamGGIN


class HeteroPReLU(nn.Module):
    def __init__(self, prelus: dict[str, nn.PReLU]) -> None:
        super(HeteroPReLU, self).__init__()
        for name, prelu in prelus.items():
            setattr(self, name, prelu)

    def forward(self, input_: dict):
        for key, value in input_.items():
            input_[key] = getattr(self, key).forward(value)
        return input_


class RGCN(DownstreamGGIN):
    def __init__(self, hidden_dim: int, output_dim: int, task: Literal["regression", "classification", "multilabel"],
                 num_layers: int = 3, batch_size: int = 32, pre_transform_args: Optional[Dict] = None, **kwargs: Any):
        """
        Implementation of the relational GCN model.

        Args:
            hidden_dim: The dimensionality of the hidden layers.
            output_dim: The output dimension of the model
            task: The type of task to perform.
            num_layers: The number of layers in the network.
            batch_size: The batch size to use
            pre_transform_args: The arguments for the pre-transforms.
            kwargs: Additional arguments
        """
        super(RGCN, self).__init__(hidden_dim, output_dim, task, num_layers, batch_size, pre_transform_args, **kwargs)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                # Set the inner layers to be a single weight without using the nodes embedding (therefore, e=-1)
                key: GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim)), eps=-1) for key in [
                    ("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")
                ]
            }))
            self.convs.append(HeteroPReLU({
                "atoms": nn.PReLU(),
                "bonds": nn.PReLU(),
                "monosacchs": nn.PReLU(),
            }))