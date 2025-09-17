from typing import Any, Literal, Optional

import torch
from torch import nn

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN


def build_mlp(input_dim: int, hidden_dim: int, num_predictions: int, size: Literal["small", "medium", "large"] = "small", dropout: float = 0.3) -> torch.nn.Module:
    if size == "small":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predictions)
        )
    elif size == "medium":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_predictions)
        )
    elif size == "large":
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_predictions)
        )


class MLP(DownstreamGGIN):
    def __init__(
            self, 
            feat_dim: int, 
            hidden_dim: int, 
            output_dim: int = 1, 
            task: Literal["regression", "classification", "multilabel", "spectrum", "spectrum"] | None = None,
            num_layers: int = 3,
            batch_size: int = 32,
            **kwargs: Any
        ):
        super().__init__(feat_dim, hidden_dim, output_dim, task, **kwargs)

        del self.convs
        del self.embedding
        if self.task is not None:
            self.head = build_mlp(
                input_dim=feat_dim,
                hidden_dim=hidden_dim,
                num_predictions=output_dim,
            )
    
    
    def forward(self, batch: HeteroDataBatch) -> dict[str, Optional[torch.Tensor]]:
        """
        Make predictions based on the molecular fingerprint.

        Args:
            batch: Batch of heterogeneous graphs.

        Returns:
            Dict holding the node embeddings (None for the MLP), the graph embedding, and the final model prediction
        """
        return {
            "node_embed": None,
            "graph_embed": batch["fp"],
            "preds": self.head(batch["fp"]) if hasattr(self, "head") else None,
        }
