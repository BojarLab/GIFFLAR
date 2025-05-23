from pathlib import Path
from typing import Literal, Any

import torch
from torch import nn, scatter_add
from torch_geometric.nn.inits import reset
from torch_geometric.utils import softmax
from torch_geometric.nn import GINConv, global_mean_pool
from torch_scatter import scatter_add
from torchmetrics import CosineSimilarity, KLDivergence, MetricCollection

from gifflar.data.utils import GlycanStorage
from gifflar.metrics import ModifiedCosineSimilarity
from gifflar.utils import get_metrics


class EmbeddingStorage(GlycanStorage):
    def __init__(self, encoder, name: str, path: str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
        """
        if not name.endswith(".pkl"):
            name += ".pkl"
        self.path = Path(path or "data") / name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self.encoder = encoder
        self.data = self._load()

    def query(self, query: str) -> torch.Tensor:
        if query not in self.data:
            try:
                self.data[query] = self.encoder(query).mean(dim=1)  # May be hard-coded to GlyLM encoders?! Encoder output has to be a tensor of token embeddings
            except Exception as e:
                print(e)
                self.data[query] = None

        return self.data[query]

    def batch_query(self, queries) -> torch.Tensor:
        results = [self.query(query) for query in queries]
        dummy = None
        for x in results:
            if x is not None:
                dummy = torch.zeros_like(x)
                break
        return torch.stack([dummy if res is None else res for res in results])

    def to(self, device):
        if hasattr(self.encoder, "to"):
            self.encoder.to(device)
        return self


class LectinStorage(EmbeddingStorage):
    def __init__(self, encoder, lectin_encoder: str, le_layer_num: int, path: str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a lectin_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        super().__init__(encoder, name=f"{lectin_encoder}_{le_layer_num}.pkl")


class MultiGlobalAttention(nn.Module):
    def __init__(self, gas: dict):
        super().__init__()
        for name, ga in gas.items():
            setattr(self, name, ga)

    def __getitem__(self, item):
        if hasattr(self, item):
            return getattr(self, item)
        return super().__getitem__(item)


class GlobalAttention(nn.Module):
    r"""Global soft attention layer from the `"Gated Graph Sequence Neural
    Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{r}_i = \sum_{n=1}^{N_i} \mathrm{softmax} \left(
        h_{\mathrm{gate}} ( \mathbf{x}_n ) \right) \odot
        h_{\mathbf{\Theta}} ( \mathbf{x}_n ),

    where :math:`h_{\mathrm{gate}} \colon \mathbb{R}^F \to
    \mathbb{R}` and :math:`h_{\mathbf{\Theta}}` denote neural networks, *i.e.*
    MLPS.

    Args:
        gate_nn (torch.nn.Module): A neural network :math:`h_{\mathrm{gate}}`
            that computes attention scores by mapping node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, 1]`, *e.g.*,
            defined by :class:`torch.nn.Sequential`.
        nn (torch.nn.Module, optional): A neural network
            :math:`h_{\mathbf{\Theta}}` that maps node features :obj:`x` of
            shape :obj:`[-1, in_channels]` to shape :obj:`[-1, out_channels]`
            before combining them with the attention scores, *e.g.*, defined by
            :class:`torch.nn.Sequential`. (default: :obj:`None`)
    """
    def __init__(self, gate_nn, nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)

    def forward(self, x, batch, size=None):
        """"""
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.int64)

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = int(batch.max()) + 1 if size is None else size

        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(gate_nn={self.gate_nn}, '
                f'nn={self.nn})')


def get_gin_layer(input_dim: int, output_dim: int) -> GINConv:
    """
    Get a GIN layer with the specified input and output dimensions

    Args:
        input_dim: The input dimension of the GIN layer
        output_dim: The output dimension of the GIN layer

    Returns:
        A GIN layer with the specified input and output dimensions
    """
    return GINConv(
        nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(output_dim),
        )
    )

def get_spectrum_prediction_head(
        input_dim: int,
        num_predictions: int,
        metric_prefix: str = "",
        size: Literal["small", "medium", "large"] = "small",
) -> tuple[nn.Module, nn.Module, dict[str, nn.Module]]:
    head = nn.Sequential(nn.Linear(input_dim, input_dim))
    for _ in range({"small": 1, "medium": 2, "large": 3}[size]):
        head.append(nn.PReLU())
        head.append(nn.Dropout(0.2))
        head.append(nn.Linear(input_dim, input_dim))
    head.append(nn.PReLU())
    head.append(nn.Dropout(0.2))
    head.append(nn.Linear(input_dim, num_predictions))
    head.append(nn.Softmax(dim=-1))

    loss = nn.CosineEmbeddingLoss()
    
    m = MetricCollection([
        CosineSimilarity(reduction="mean"),
        KLDivergence(reduction="mean", log_prob=True),
        ModifiedCosineSimilarity(num_bins=2048, tolerance=0.2),
    ])
    metrics = {
        "train": m.clone(prefix="train/" + metric_prefix), 
        "val": m.clone(prefix="val/" + metric_prefix),
        "test": m.clone(prefix="test/" + metric_prefix)
    }
    return head, loss, metrics


def get_prediction_head(
        input_dim: int,
        num_predictions: int,
        task: Literal["regression", "classification", "multilabel", "spectrum"],
        metric_prefix: str = "",
        size: Literal["small", "medium", "large"] = "small",
) -> tuple[nn.Module, nn.Module, dict[str, nn.Module]]:
    """
    Create the prediction head for the specified dimensions and generate the loss and metrics for the task

    Args:
        input_dim: The input dimension of the prediction head
        num_predictions: The number of predictions to make
        task: The task to perform, either "regression", "classification", "multilabel", "spectrum"
        metric_prefix: The prefix to use for the metrics

    Returns:
        A tuple containing the prediction head, the loss function and the metrics for the task
    """
    if task == "spectrum":
        return get_spectrum_prediction_head(input_dim, num_predictions, metric_prefix, size)
    # Create the prediction head
    if size == "small":
        head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, num_predictions)
        )
    elif size == "medium":
        head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 4, num_predictions)
        )
    elif size == "large":
        head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 4, num_predictions)
        )

    # Create the loss function based on the task and the number of outputs to predict
    if task == "regression":
        loss = nn.MSELoss()
    elif num_predictions == 1 or task == "multilabel":
        loss = nn.BCEWithLogitsLoss()
    else:
        # Add a softmax layer if the task is classification and there are multiple predictions
        head.append(nn.Softmax(dim=-1))
        loss = nn.CrossEntropyLoss()

    # Get the metrics for the task
    metrics = get_metrics(task, num_predictions, metric_prefix)
    return head, loss, metrics


class MultiEmbedding(nn.Module):
    """Class storing multiple embeddings in a dict-format and allowing for training them as nn.Module"""

    def __init__(self, embeddings: dict[str, nn.Embedding]):
        """
        Initialize the MultiEmbedding class

        Args:
            embeddings: A dictionary of embeddings to store
        """
        super().__init__()
        for name, embedding in embeddings.items():
            setattr(self, name, embedding)

    def forward(self, input_: dict[str, Any], name: str) -> torch.Tensor:
        """
        Embed the input using the specified embedding

        Args:
            input_: The input to the embedding
            name: The name of the embedding to use
        """
        return getattr(self, name).forward(input_)


class GIFFLARPooling(nn.Module):
    """Class to perform pooling on the output of the GIFFLAR model"""

    def __init__(self, mode: str = "global_mean"):
        """
        Initialize the GIFFLARPooling class

        Args:
            mode: The pooling mode to use, either
                - global_mean:
                    Standard mean pooling over all nodes in the graph
                - local_mean:
                    Standard mean pooling over all nodes of each type, then mean over the node types
                - weighted_mean:
                    Local mean pooling with learnable weights per node type in the second mean-computation
                - global_attention:
                    Standard self-attention mechanism over all nodes in the graph
                - local_attention:
                    Standard self-attention mechanism over all nodes of each type, then mean over the nodes types
                - weighted_attention:
                    Standard self-attention mechanism over all nodes of each type, then mean with learnable weights
                        per node type in the cell aggregation.
                - cell_attention:
                    Local mean pooling with self-attention over the cell results
                - local_cell_attention:
                    Local mean pooling with self-attention over the cell results of each type, then mean over the cell types
                - weighted_cell_attention:
                    Local mean pooling with self-attention over the cell results (their means) and learnable weights for aggregation
        """
        super().__init__()
        self.mode = mode
        self.attention = None
        self.weights = None
        if "weighted" in mode:
            print("Create weighting")
            self.weights = nn.Parameter(torch.ones(3), requires_grad=True)
        if "attention" in mode:
            print("Create attention")
            if mode == "global_attention":
                self.attention = MultiGlobalAttention({"": GlobalAttention(
                    gate_nn=nn.Linear(1024, 1),
                    nn=nn.Sequential(
                        nn.Linear(1024, 1024),
                        nn.PReLU(),
                        nn.Dropout(0.2),
                        nn.BatchNorm1d(1024),
                        nn.Linear(1024, 1024),
                    ),
                )})
            else:
                self.attention = MultiGlobalAttention({key: GlobalAttention(
                    gate_nn=nn.Linear(1024, 1),
                    nn=nn.Sequential(
                        nn.Linear(1024, 1024),
                        nn.PReLU(),
                        nn.Dropout(0.2),
                        nn.BatchNorm1d(1024),
                        nn.Linear(1024, 1024),
                    ),
                ) for key in ["atoms", "bonds", "monosacchs"]})

    def to(self, device):
        if self.weights is not None:
            self.weights = self.weights.to(device)
        if self.attention is not None:
            if self.mode == "global_attention":
                self.attention[""].to(device)
            else:
                self.attention["atoms"].to(device)
                self.attention["bonds"].to(device)
                self.attention["monosacchs"].to(device)
        super(GIFFLARPooling, self).to(device)
        return self

    def forward(self, nodes: dict[str, torch.Tensor], batch_ids: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform pooling on the input

        Args:
            x: The input to the pooling layer
        """
        device = nodes["atoms"].device
        match self.mode:
            case "global_mean":
                return global_mean_pool(
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([batch_ids["atoms"], batch_ids["bonds"], batch_ids["monosacchs"]], dim=0)
                )
            case "local_mean":
                nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) for key in nodes.keys()}
                return global_mean_pool(
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([torch.arange(len(nodes[key])).to(device) for key in ["atoms", "bonds", "monosacchs"]], dim=0)
                )
            case "weighted_mean":
                nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
                return global_mean_pool(
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([torch.arange(len(nodes[key])).to(device) for key in ["atoms", "bonds", "monosacchs"]], dim=0)
                )
            case "global_attention":
                return self.attention[""](
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([batch_ids["atoms"], batch_ids["bonds"], batch_ids["monosacchs"]], dim=0)
                )
            case "local_attention":
                nodes = {key: self.attention[key](nodes[key], batch_ids[key]) for key in nodes.keys()}
                return global_mean_pool(
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([torch.arange(len(nodes[key])).to(device) for key in ["atoms", "bonds", "monosacchs"]], dim=0)
                )
            case "weighted_attention":
                nodes = {key: self.attention[key](nodes[key], batch_ids[key]) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
                device = nodes["atoms"].device
                return global_mean_pool(
                    torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
                    torch.concat([torch.arange(len(nodes[key])).to(device) for key in ["atoms", "bonds", "monosacchs"]], dim=0)
                )
            #case "cell_attention":
            #    nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) for key in nodes.keys()}
            #    return self.attention[""](
            #        torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            #        torch.concat([torch.arange(len(nodes["atoms"])), torch.arange(len(nodes["bonds"])), torch.arange(len(nodes["monosacchs"]))], dim=0)
            #    )
            #case "local_cell_attention":
            #    nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) for key in nodes.keys()}
            #    return global_mean_pool(
            #        torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            #        torch.concat([torch.arange(len(nodes["atoms"])), torch.arange(len(nodes["bonds"])), torch.arange(len(nodes["monosacchs"]))], dim=0)
            #    )
            #case "weighted_cell_attention":
            #    nodes = {key: global_mean_pool(nodes[key], batch_ids[key]) * self.weights[i] for i, key in enumerate(["atoms", "bonds", "monosacchs"])}
            #    return global_mean_pool(
            #        torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            #        torch.concat([torch.arange(len(nodes["atoms"])), torch.arange(len(nodes["bonds"])), torch.arange(len(nodes["monosacchs"]))], dim=0)
            #    )
            #case "double_attention":
            #    nodes = {key: self.attention[key](global_mean_pool(nodes[key], batch_ids[key])) for key in nodes.keys()}
            #    return global_mean_pool(
            #        torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            #        torch.concat([torch.arange(len(nodes["atoms"])), torch.arange(len(nodes["bonds"])), torch.arange(len(nodes["monosacchs"]))], dim=0)
            #    )
            #case "weighted_double_attention":
            #    nodes = {key: self.attention[key](global_mean_pool(nodes[key], batch_ids[key])) for key in nodes.keys()}
            #    return global_mean_pool(
            #        torch.concat([nodes["atoms"], nodes["bonds"], nodes["monosacchs"]], dim=0),
            #        torch.concat([torch.arange(len(nodes["atoms"])), torch.arange(len(nodes["bonds"])), torch.arange(len(nodes["monosacchs"]))], dim=0)
            #    )
            case _:
                raise ValueError(f"Pooling method {self.mode} not supported.")

