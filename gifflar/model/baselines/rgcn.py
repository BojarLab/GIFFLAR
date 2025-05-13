from typing import Literal, Optional, Any

import torch
from torch import nn
from torch_geometric.nn import HeteroConv, GINConv, RGCNConv, global_mean_pool

from gifflar.data.hetero import HeteroDataBatch
from gifflar.model.downstream import DownstreamGGIN


class RGCN(DownstreamGGIN):
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            output_dim: int,
            task: Literal["regression", "classification", "multilabel", "spectrum"],
            num_layers: int = 3,
            batch_size: int = 32,
            pre_transform_args: Optional[dict] = None,
            **kwargs: Any,
    ):
        super(RGCN, self).__init__(feat_dim, hidden_dim, output_dim, task, num_layers, batch_size,
                                          pre_transform_args, **kwargs)

        dims = [feat_dim]
        if feat_dim <= hidden_dim // 2:
            dims += [hidden_dim // 2]
        else:
            dims += [hidden_dim]
        dims += [hidden_dim] * (num_layers - 1)
        self.convs = torch.nn.ModuleList([
            RGCNConv(dims[i], dims[i + 1], num_relations=5) for i in range(num_layers)
        ])
        
        del self.pooling
        self.pooling = global_mean_pool  # GIFFLARPooling()


    def forward(self, batch: HeteroDataBatch) -> dict[str, torch.Tensor]:
        """
        Compute the node embeddings.

        Args:
            batch: The batch of data to process

        Returns:
            node_embed: The node embeddings
        """

        node_embeds = [torch.stack([self.embedding.forward(batch["rgcn_x"][i], batch["rgcn_node_type"][i]) for i in range(len(batch["rgcn_x"]))])]
        for pe in self.addendum:
            node_embeds.append(batch[f"rgcn_{pe}"])
        node_embeds = torch.concat(node_embeds, dim=1)

        for conv in self.convs:
            node_embeds = conv(node_embeds, batch["rgcn_edge_index"], batch["rgcn_edge_type"])
        
        graph_embed = self.pooling(node_embeds, batch["rgcn_batch"])
        pred = self.head(graph_embed)
        return {
                "node_embed": node_embeds,
            "graph_embed": graph_embed,
            "preds": pred,
        }

