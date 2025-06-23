import copy
from pathlib import Path
from typing import Any, Union, Literal

import torch
from glycowork.glycan_data.loader import lib, HashableDict
from glycowork.motif.graph import glycan_to_nxGraph
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE
from torch_geometric.transforms.base_transform import BaseTransform
from torch_geometric.utils import from_networkx, to_dense_adj
from tqdm import tqdm

from gifflar.model.utils import GIFFLARPooling
from gifflar.utils import bond_map, lib_map, atom_map, mono_map, get_mods_list


def split_hetero_graph(data: HeteroData) -> tuple[Data, Data, Data]:
    """
    Split a heterogeneous graph into homogeneous graphs.

    Args:
        data: The heterogeneous graph.

    Returns:
        A tuple of three homogeneous graphs, one for atoms, one for bonds, and one for monosaccharides.
    """
    atoms_data = Data(
        edge_index=data["atoms", "coboundary", "atoms"]["edge_index"],
        num_nodes=data["atoms"]["num_nodes"],
    )
    bond_data = Data(
        edge_index=data["bonds", "boundary", "bonds"]["edge_index"],
        num_nodes=data["bonds"]["num_nodes"],
    )
    monosacchs_data = Data(
        edge_index=data["monosacchs", "boundary", "monosacchs"]["edge_index"],
        num_nodes=data["monosacchs"]["num_nodes"],
    )
    return atoms_data, bond_data, monosacchs_data


def hetero_to_homo(data: HeteroData) -> Data:
    """Convert a heterogeneous graph to a homogeneous by collapsing the node types and removing all node features."""
    bond_edge_index = data["bonds", "boundary", "bonds"]["edge_index"] + data["atoms"]["num_nodes"]
    monosacchs_edge_index = data["monosacchs", "boundary", "monosacchs"]["edge_index"] + data["atoms"]["num_nodes"] + \
                            data["bonds"]["num_nodes"]
    return Data(
        edge_index=torch.cat([
            data["atoms", "coboundary", "atoms"]["edge_index"],
            bond_edge_index, monosacchs_edge_index
        ], dim=1),
        num_nodes=data["atoms"]["num_nodes"] + data["bonds"]["num_nodes"] + data["monosacchs"]["num_nodes"]
    )


class RootTransform(BaseTransform):
    """Root transformation class."""

    def __init__(self, **kwargs: Any):
        pass


class GIFFLARTransform(RootTransform):
    """Transformation to bring data into a GIFFLAR format"""

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Transform the data into a GIFFLAR format. This means to compute the simplex network and create a heterogenous
        graph from it.

        Args:
            data: The input data.

        Returns:
            The transformed data.
        """
        # Set up the atom information
        data["atoms"].x = torch.tensor([
            atom_map.get(atom.GetAtomicNum(), 1) for atom in data["mol"].GetAtoms()
        ])
        data["atoms"].num_nodes = len(data["atoms"].x)

        # prepare all data that can be extracted from one iteration over all bonds
        data["bonds"].x = []
        data["atoms", "coboundary", "atoms"].edge_index = []
        data["atoms", "to", "bonds"].edge_index = []
        data["bonds", "to", "monosacchs"].edge_index = []

        # fill all bond-related information
        for bond in data["mol"].GetBonds():
            data["bonds"].x.append(bond_map.get(bond.GetBondDir(), 1))
            data["atoms", "coboundary", "atoms"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx())
            ]
            data["atoms", "to", "bonds"].edge_index += [
                (bond.GetBeginAtomIdx(), bond.GetIdx()),
                (bond.GetEndAtomIdx(), bond.GetIdx())
            ]
            data["bonds", "to", "monosacchs"].edge_index.append(
                (bond.GetIdx(), bond.GetIntProp("mono_id"))
            )

        # transform the data into tensors
        data["bonds"].x = torch.tensor(data["bonds"].x)
        data["bonds"].num_nodes = len(data["bonds"].x)
        data["atoms", "coboundary", "atoms"].edge_index = torch.tensor(data["atoms", "coboundary", "atoms"].edge_index,
                                                                       dtype=torch.long).T
        data["atoms", "to", "bonds"].edge_index = torch.tensor(data["atoms", "to", "bonds"].edge_index,
                                                               dtype=torch.long).T
        data["bonds", "to", "monosacchs"].edge_index = torch.tensor(data["bonds", "to", "monosacchs"].edge_index,
                                                                    dtype=torch.long).T

        # compute both types of linkages between bonds
        data["bonds", "boundary", "bonds"].edge_index = torch.tensor(
            [(bond1.GetIdx(), bond2.GetIdx()) for atom in data["mol"].GetAtoms() for bond1 in atom.GetBonds()
             for bond2 in atom.GetBonds() if bond1.GetIdx() != bond2.GetIdx()], dtype=torch.long).T
        data["bonds", "coboundary", "bonds"].edge_index = torch.tensor(
            [(bond1, bond2) for ring in data["mol"].GetRingInfo().BondRings() for bond1 in ring for bond2 in ring if
             bond1 != bond2], dtype=torch.long).T

        # Set up the monosaccharide information
        data["monosacchs"].x = torch.tensor([  # This does not make sense. The monomer-ids are categorical features
            lib_map.get(data["tree"].nodes[node]["name"], 1) for node in data["tree"].nodes
        ])
        data["monosacchs"].num_nodes = len(data["monosacchs"].x)
        data["monosacchs", "boundary", "monosacchs"].edge_index = []
        for a, b in data["tree"].edges:
            data["monosacchs", "boundary", "monosacchs"].edge_index += [(a, b), (b, a)]
        data["monosacchs", "boundary", "monosacchs"].edge_index = torch.tensor(
            data["monosacchs", "boundary", "monosacchs"].edge_index, dtype=torch.long).T

        return data


class RGCNTransform(RootTransform):
    def __call__(self, data: HeteroData) -> HeteroData:
        data["rgcn_x"] = torch.cat([
            data["atoms"].x,
            data["bonds"].x,
            data["monosacchs"].x
        ])
        data["rgcn_node_type"] = ["atoms"] * len(data["atoms"].x) + ["bonds"] * len(data["bonds"].x) + ["monosacchs"] * len(data["monosacchs"].x)
        data["rgcn_num_nodes"] = len(data["rgcn_x"])
        edges = [0] * data["atoms", "coboundary", "atoms"].edge_index.shape[1]
        edges += [1] * data["atoms", "to", "bonds"].edge_index.shape[1]
        edges += [2] * data["bonds", "boundary", "bonds"].edge_index.shape[1]
        edges += [3] * data["bonds", "to", "monosacchs"].edge_index.shape[1]
        try:
            edges += [4] * data["monosacchs", "boundary", "monosacchs"].edge_index.shape[1]
        except:
            pass
        data["rgcn_edge_type"] = torch.tensor(edges)
        
        tmp = []
        offset = {"atoms": 0,
                  "bonds": data["atoms"]["num_nodes"],
                  "monosacchs": data["atoms"]["num_nodes"] + data["bonds"]["num_nodes"]}
        for key in [("atoms", "coboundary", "atoms"),
                    ("atoms", "to", "bonds"),
                    ("bonds", "to", "monosacchs"),
                    ("bonds", "boundary", "bonds"),
                    ("monosacchs", "boundary", "monosacchs")]:
            try:
                tmp.append(torch.stack([
                    data[key].edge_index[0] + offset[key[0]],
                    data[key].edge_index[1] + offset[key[2]],
                ]))
            except:
                pass
        data["rgcn_edge_index"] = torch.cat(tmp, dim=1)
        return data


class GNNGLYTransform(RootTransform):
    """Transformation to bring data into a GNNGLY format"""

    def __init__(self, **kwargs: Any):
        """
        Initialize the GNNGLY transformation by setting up the one-hot encoders.

        Args:
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.atom_encoder = torch.eye(101)
        self.chiral_encoder = torch.eye(4)
        self.degree_encoder = torch.eye(13)
        self.charge_encoder = torch.eye(5)
        self.h_encoder = torch.eye(5)
        self.hybrid_encoder = torch.eye(5)

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Transform the data into a GNNGLY format.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        data["gnngly_x"] = torch.stack([torch.concat([
            self.atom_encoder[min(atom.GetAtomicNum(), 100)],
            self.chiral_encoder[min(atom.GetChiralTag(), 3)],
            self.degree_encoder[min(atom.GetDegree(), 12)],
            self.charge_encoder[min(atom.GetFormalCharge(), 4)],
            self.h_encoder[min(atom.GetTotalNumHs(), 4)],
            self.hybrid_encoder[min(atom.GetHybridization(), 4)],
        ]) for atom in data["mol"].GetAtoms()])
        data["gnngly_num_nodes"] = len(data["gnngly_x"])
        data["gnngly_edge_index"] = copy.deepcopy(data["atoms", "coboundary", "atoms"].edge_index)
        return data


class ECFPTransform(RootTransform):
    """Transformation to bring data into an ECFP format for MLP and SL model usage"""

    def __init__(self, **kwargs: Any):
        """
        Initialize the ECFP transformation.

        Args:
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Transform the data into an ECFP format.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        # Check if the RDKit fingerprint generator is available
        data["fp"] = torch.tensor(self.ecfp.GetFingerprint(data["mol"]), dtype=torch.float).reshape(1, -1)
        return data


class SweetNetTransform(RootTransform):
    def __init__(self, glycan_lib: dict[str, int] = lib, **kwargs: Any):
        """
        Transformation to convert a glycan IUPAC string to a PyG Data object for the SweetNet model.

        Args:
            glycan_lib: The glycan library to use.
            kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        self.glycan_lib = HashableDict(glycan_lib)

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Transform the data into a SweetNet format.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        try:
            d = from_networkx(glycan_to_nxGraph(data["IUPAC"], self.glycan_lib))
        except:
            d = from_networkx(glycan_to_nxGraph("Gal", self.glycan_lib))
        data["sweetnet_x"] = d["labels"].clone().detach().reshape(-1, 1)
        data["sweetnet_num_nodes"] = len(data["sweetnet_x"])
        data["sweetnet_edge_index"] = d.edge_index
        return data


class LaplacianPE(AddLaplacianEigenvectorPE):
    attr_name = "lap_pe"

    def __init__(self, dim: int, individual: bool = True, **kwargs: Any):
        """
        Initialize the Laplacian positional encoding transformation.

        Args:
            dim: Number of eigenvectors to compute. (k)
            individual: Whether to compute eigenvectors for each node type individually.
            kwargs: Additional arguments.
        """
        super().__init__(k=dim, attr_name=self.attr_name, **kwargs)
        self.max_dim = dim
        self.individual = individual

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Compute the Laplacian eigenvectors for the input data.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        if self.individual:  # compute the laplacian eigenvectors for each node type individually
            for d, name in zip(split_hetero_graph(data), ["atoms", "bonds", "monosacchs"]):
                if d["num_nodes"] <= self.max_dim:
                    self.k = min(d["num_nodes"] - 1, self.max_dim)
                    if self.k == 0:
                        d[self.attr_name] = torch.tensor([[]])
                    else:
                        d = super(LaplacianPE, self).forward(d)
                    pad = torch.zeros(d["num_nodes"], self.max_dim - d[self.attr_name].size(1))
                    d[self.attr_name] = torch.cat([d[self.attr_name], pad], dim=1)
                    self.k = self.max_dim
                else:
                    d = super(LaplacianPE, self).forward(d)
                data[f"{name}_{self.attr_name}"] = d[self.attr_name]
        else:  # or for the whole graph
            d = hetero_to_homo(data)
            if self.k == 0:
                d[self.attr_name] = torch.tensor([[]])
            else:
                d = super(LaplacianPE, self).forward(d)
            data[f"atoms_{self.attr_name}"] = d[self.attr_name][:data["atoms"]["num_nodes"]]
            data[f"bonds_{self.attr_name}"] = d[self.attr_name][
                                              data["atoms"]["num_nodes"]:-data["monosacchs"]["num_nodes"]]
            data[f"monosacchs_{self.attr_name}"] = d[self.attr_name][-data["monosacchs"]["num_nodes"]:]

        if "rgcn_x" in data:
            data[f"rgcn_{self.attr_name}"] = super(LaplacianPE, self).forward(
                Data(x=data["rgcn_x"], edge_index=data["rgcn_edge_index"], num_nodes=len(data["rgcn_x"]))
            )[self.attr_name]

        return data


class RandomWalkPE(RootTransform):
    """Random walk dense version."""
    attr_name = "rw_pe"

    def __init__(self, dim: int, individual: bool = True, cuda: bool = False, **kwargs: Any):
        """
        Initialize the RandomWalk positional encoding.

        Args:
            dim: The number of random walk steps (walk_length).
            individual: Whether to compute eigenvectors for each node type individually.
            cuda: Whether to move the computation to GPU or not
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.walk_length = dim
        self.individual = individual
        self.cuda = cuda

    def forward(self, data: Data) -> Data:
        """
        Compute the random walk positional encodings for the input data.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        if data["num_nodes"] == 1:
            data[self.attr_name] = torch.zeros(1, 20)
            return data
        adj = to_dense_adj(data.edge_index, max_num_nodes=data["num_nodes"]).squeeze(0)
        row_sums = adj.sum(dim=1, keepdim=True)
        adj = adj / row_sums.clamp(min=1)
        pe_list = [torch.zeros(adj.shape[0])]
        walk_matrix = adj
        for _ in range(self.walk_length - 1):
            walk_matrix = walk_matrix @ adj
            pe_list.append(walk_matrix.diag())
        pe = torch.stack(pe_list, dim=-1)
        data[self.attr_name] = pe
        return data

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Compute the random walk positional encodings for the input data.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        if self.individual:  # compute the random walk positional encodings for each node type individually
            for d, name in zip(split_hetero_graph(data), ["atoms", "bonds", "monosacchs"]):
                d = self.forward(d)
                data[f"{name}_{self.attr_name}"] = d[self.attr_name]
        else:  # or for the whole graph
            d = hetero_to_homo(data)
            d = self.forward(d)
            data[f"atoms_{self.attr_name}"] = d[self.attr_name][:data["atoms"]["num_nodes"]]
            data[f"bonds_{self.attr_name}"] = d[self.attr_name][
                                              data["atoms"]["num_nodes"]:-data["monosacchs"]["num_nodes"]]
            data[f"monosacchs_{self.attr_name}"] = d[self.attr_name][-data["monosacchs"]["num_nodes"]:]
        if "rgcn_x" in data:
            data[f"rgcn_{self.attr_name}"] = self.forward(
                Data(x=data["rgcn_x"], edge_index=data["rgcn_edge_index"], num_nodes=len(data["rgcn_x"]))
            )[self.attr_name]
        return data


class MonosaccharidePrediction(RootTransform):
    def __init__(self, mode: Literal["mono", "mods", "both"], **kwargs: Any):
        """
        Add labels for monosaccharide-prediction to the data.

        Params:
            mode: which tasks to add:
                mono: Only monomer-prediction
                mods: Only prediction of monosaccharides
                both: mono + mods
            kwargs: Additional arguments
        """
        super(MonosaccharidePrediction, self).__init__(**kwargs)
        self.mode = mode

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Add mono_y and mods_y to the data.

        Params:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        if self.mode in {"mono", "both"}:
            data["mono_y"] = torch.tensor([
                mono_map.get(data["tree"].nodes[x]["name"], 0) for x in range(data["monosacchs"].num_nodes)
            ])
        if self.mode in {"mods", "both"}:
            data["mods_y"] = torch.tensor([
                get_mods_list(data["tree"].nodes[x]) for x in range(data["monosacchs"].num_nodes)
            ])
        return data


class PretrainEmbed(RootTransform):
    """Run a GIFFLAR model to embed the input data."""

    def __init__(self, folder: str, dataset_name: str, model_name: str, pooling: str, hash_str: str, **kwargs: Any):
        """
        Set up the Embedding from a pretrained model.

        Args:
            folder: The folder where the preprocessed data is stored.
            dataset_name: The name of the dataset.
            model_name: The name of the model.
            hash_str: The hash string of the model.
            kwargs: Additional arguments
        """
        super(PretrainEmbed, self).__init__(**kwargs)
        self.data_dir = Path(folder) / f"{dataset_name}_{model_name}_{hash_str}"
        self.pooling = GIFFLARPooling()
        self.layer = -1

    def __call__(self, data: HeteroData) -> HeteroData:
        """
        Compute the embeddings for the input data.

        Args:
            data: The input data to be transformed.

        Returns:
            The transformed data.
        """
        h = hash(data["smiles"])
        p = self.data_dir / f"{h}.pt"
        if not p.exists():
            print(data["smiles"], "not found in preprocessed data.")
            data["fp"] = torch.zeros_like(data["fp"])
            return data

        # apply the masks and extract the node embeddings and compute batch ids
        node_embeds = torch.load(p)[self.layer]
        batch_ids = {key: torch.zeros(len(node_embeds[key]), dtype=torch.long) for key in
                     ["atoms", "bonds", "monosacchs"]}
        data["fp"] = self.pooling(node_embeds, batch_ids)
        return data


class TQDMCompose(Compose):
    """Add TQDM bars to the transformations calculated in this Compose object"""

    def forward(self, data: list[Union[Data, HeteroData]]):
        """
        Apply transformation in order to the input data and keep TQDM bars for process tracking.

        Args:
            data: The data to be transformed

        Returns:
            The transformed data
        """
        if len(data) == 0 or len(self.transforms) == 0:
            return data
        for transform in tqdm(self.transforms, desc=f"TQDM Transform"):
            if not isinstance(data, (list, tuple)):
                output = transform(data)
            else:
                output = []
                for d in tqdm(data, desc=str(transform), leave=False):
                    if isinstance(d, (Data, HeteroData)):
                        output.append(transform(d))
                    elif isinstance(d, (list, tuple)):
                        output.append(tuple(transform(dd) for dd in d))
                    else:
                        raise ValueError(f"Unsupported data type: {type(d)}")
        return output


def get_pretransforms(dataset_name: str = "", **pre_transform_args: dict[str, dict]) -> TQDMCompose:
    """
    Calculate the list of pre-transforms to be applied to the data.

    Args:
        dataset_name: The name of the dataset currently being used.
        pre_transform_args: The arguments for the pre-transforms.

    Returns:
        A TQDMCompose object containing the pre-transforms.
    """
    pre_transforms = []
    for name, args in pre_transform_args.items():
        match (name):
            case "GIFFLARTransform":
                pre_transforms.append(GIFFLARTransform(**args or {}))
            case "GNNGLYTransform":
                pre_transforms.append(GNNGLYTransform(**args or {}))
            case "ECFPTransform":
                pre_transforms.append(ECFPTransform(**args or {}))
            case "SweetNetTransform":
                pre_transforms.append(SweetNetTransform(**args or {}))
            case "RGCNTransform":
                pre_transforms.append(RGCNTransform(**args or {}))
            case "LaplacianPE":
                pre_transforms.append(LaplacianPE(**args or {}))
            case "RandomWalkPE":
                pre_transforms.append(RandomWalkPE(**args or {}))
            case "MonosaccharidePrediction":
                pre_transforms.append(MonosaccharidePrediction(**args or {}))
            case "PretrainEmbed":
                pre_transforms.append(PretrainEmbed(dataset_name=dataset_name, **args or {}))
            case _:
                pass

    return TQDMCompose(pre_transforms)
