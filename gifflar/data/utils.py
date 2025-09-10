import copy
import pickle
from pathlib import Path
from typing import Optional

import glyles
import networkx as nx
from glyles.glycans.factory.factory import MonomerFactory
from glyles.glycans.poly.merger import Merger
from rdkit import Chem
from torch_geometric.data import HeteroData
from rdkit.Chem import rdDepictor, BondType


def nx2mol(G: nx.Graph, sanitize=True) -> Chem.Mol:
    """
    Convert a molecules from a networkx.Graph to RDKit.

    Args:
        G: The graph representing a molecule
        sanitize: A bool flag indicating to sanitize the resulting molecule (should be True for "production mode" and
            False when debugging this function)

    Returns:
        The converted, sanitized molecules in RDKit represented by the input graph
    """
    # Create the molecule
    mol = Chem.RWMol()

    # Extract the node attributes
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    mono_ids = nx.get_node_attributes(G, 'mono_id')

    # Create all atoms based on their representing nodes
    node_to_idx = {}
    for node in G.nodes():
        a = Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetIntProp("mono_id", mono_ids[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    # Extract the edge attributes
    bond_types = nx.get_edge_attributes(G, 'bond_type')
    mono_bond_ids = nx.get_edge_attributes(G, 'mono_id')

    # Connect the atoms based on the edges from the graph
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mono_idx = mono_bond_ids[first, second]
        idx = mol.AddBond(ifirst, isecond, bond_type) - 1
        mol.GetBondWithIdx(idx).SetIntProp("mono_id", mono_idx)

    # print(Chem.MolToSmiles(mol))
    if sanitize:
        Chem.SanitizeMol(mol)
    return mol


def clean_tree(tree: nx.Graph) -> Optional[nx.Graph]:
    """
    Clean the tree from unnecessary node features and store only the IUPAC name.

    Args:
        tree: The tree to clean

    Returns:
        The cleaned tree
    """
    for node in tree.nodes:
        attributes = copy.deepcopy(tree.nodes[node])
        if "type" in attributes and isinstance(attributes["type"], glyles.glycans.mono.monomer.Monomer):
            tree.nodes[node].clear()
            tree.nodes[node].update({"iupac": "".join([x[0] for x in attributes["type"].recipe]),
                                     "name": attributes["type"].name, "recipe": attributes["type"].recipe})
        else:
            return None
    return tree


def iupac2mol(iupac: str) -> Optional[HeteroData]:
    """
    Convert a glycan stored given as IUPAC-condensed string into an RDKit molecule while keeping the information of
    which atom and which bond belongs to which monosaccharide.

    Args:
        iupac: The IUPAC-condensed string of the glycan to convert

    Returns:
        A HeteroData object containing the IUPAC string, the SMILES representation, the RDKit molecule, and the
        monosaccharide tree
    """
    if "{" in iupac or "?" in iupac:
        return None

    # convert the IUPAC string using GlyLES
    glycan = glyles.Glycan(iupac)
    # print(glycan.get_smiles())

    # get it's underlying monosaccharide-tree
    tree = glycan.parse_tree

    # re-merge the monosaccharide tree using networkx graphs to keep the assignment of atoms and bonds to monosacchs.
    _, merged = Merger(MonomerFactory()).merge(tree, glycan.root_orientation, glycan.start, smiles_only=False)
    mol = nx2mol(merged)

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())

    smiles = Chem.MolToSmiles(mol)
    if len(smiles) < 10 or not isinstance(tree, nx.Graph):
        return None

    tree = clean_tree(tree)

    data = HeteroData()
    data["IUPAC"] = iupac
    data["smiles"] = smiles
    data["mol"] = mol
    data["tree"] = tree
    return data


class GlycanStorage:
    def __init__(self, path: Path | str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a glycan_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        self.path = Path(path or "data") / "glycan_storage.pkl"

        # Fill the storage from the file
        self.data = self._load()

    def close(self) -> None:
        """
        Close the storage by storing the dictionary at the location provided at initialization.
        """
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac: str) -> Optional[HeteroData]:
        """
        Query the storage for a IUPAC string.

        Args:
            iupac: The IUPAC string of the query glycan

        Returns:
            A HeteroData object corresponding to the IUPAC string or None, if the IUPAC string could not be processed.
        """
        if iupac not in self.data:
            try:
                self.data[iupac] = iupac2mol(iupac)
            except Exception as e:
                self.data[iupac] = None
        return copy.deepcopy(self.data[iupac])

    def _load(self) -> dict[str, Optional[HeteroData]]:
        """
        Load the internal dictionary from the file, if it exists, otherwise, return an empty dict.

        Returns:
            The loaded (if possible) or an empty dictionary
        """
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}
