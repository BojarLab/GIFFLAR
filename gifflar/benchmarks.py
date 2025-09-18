import copy
import pickle
import urllib.request
from pathlib import Path
from typing import Literal, Optional

import glyles
import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species as taxonomy
from glycowork.motif.processing import canonicalize_iupac
from tqdm import tqdm
from nxontology.imports import from_file
from glycowork.glycan_data.loader import df_species, df_glycan, build_custom_df
import networkx as nx
from datasail.sail import datasail

from gifflar.utils import iupac2smiles


MIN_FREQUENCY = 15
UBERON_ROOTS = [
    "UBERON:0000045", "UBERON:0000476", "UBERON:0003102", "UBERON:0004119",
    "UBERON:0004120", "UBERON:0004121", "UBERON:0005090", "UBERON:0005162",
    "UBERON:0005389", "UBERON:0009856", "UBERON:0010000", "UBERON:0010314",
    "UBERON:0015212", "UBERON:0034768", "UBERON:0000173", "UBERON:0000174",
    "UBERON:0000456", "UBERON:0001011", "UBERON:0001968", "UBERON:0006314",
    "UBERON:0006535", "UBERON:0008944", "UBERON:0022293", "UBERON:0002050",
    "CL:0000039", "CL:0000151", "CL:0000211", "CL:0000219",
    "CL:0000225", "CL:0000325", "CL:0000329", "CL:0000413",
    "CL:0000988", "CL:0002242", "CL:0011115"
]
UBERON_ROOT_SET = set(UBERON_ROOTS)
UBERON_ROOT_MAP = {root: r for r, root in enumerate(UBERON_ROOTS)}


class SMILESStorage:
    def __init__(self, path: Path | str | None = None):
        """
        Initialize the wrapper around a dict.

        Args:
            path: Path to the directory. If there's a glycan_storage.pkl, it will be used to fill this object,
                otherwise, such file will be created.
        """
        self.path = Path(path or "data") / "smiles_storage.pkl"

        # Fill the storage from the file
        self.data = self._load()

    def close(self) -> None:
        """
        Close the storage by storing the dictionary at the location provided at initialization.
        """
        with open(self.path, "wb") as out:
            pickle.dump(self.data, out)

    def query(self, iupac: str) -> Optional[str]:
        """
        Query the storage for a IUPAC string.

        Args:
            iupac: The IUPAC string of the query glycan

        Returns:
            A HeteroData object corresponding to the IUPAC string or None, if the IUPAC string could not be processed.
        """
        if iupac not in self.data:
            if "{" in iupac or "?" in iupac:
                self.data[iupac] = None
                return None
            try:
                translation = glyles.convert(iupac)
                self.data[iupac] = translation[0][1] if isinstance(translation, list) else translation
            except Exception:
                self.data[iupac] = None
        return copy.deepcopy(self.data[iupac])

    def _load(self) -> dict[str, Optional[str]]:
        """
        Load the internal dictionary from the file, if it exists, otherwise, return an empty dict.

        Returns:
            The loaded (if possible) or an empty dictionary
        """
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return {}


def to_oh(names):
    vec = np.zeros(len(UBERON_ROOTS))
    for name in names:
        vec[UBERON_ROOT_MAP[name]] = 1
    return vec


def standardize_iupac(iupac):
    try:
        return canonicalize_iupac(iupac)
    except:
        return None


def get_taxonomic_level(
        root: Path,
        level: Literal["Domain", "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
) -> Path:
    """
    Extract taxonomy data at a specific level, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.
        level: The taxonomic level to extract the data from.

    Returns:
        Path to the TSV file storing the processed dataset.
    """
    if not (p := (root / Path(f"taxonomy_{level}.tsv"))).exists():
        # Read in the taxonomy data
        tax = taxonomy[["glycan", level]]

        # Standardize the IUPAC names
        tax["IUPAC"] = tax["glycan"].apply(standardize_iupac)
        tax = tax[tax["IUPAC"].notna()]

        # Remove all glycans that do not have a valid taxonomic level
        tax = tax[tax[level] != "undetermined"]
        
        # One-hot encode the individual classes and collate them for glycans that are the same
        tax = pd.concat([tax["IUPAC"], pd.get_dummies(tax[level])], axis=1)
        tax = tax.groupby('IUPAC').agg("sum").reset_index()

        # Ensure onehot values for classes are 0 or 1
        classes = [x for x in tax.columns if x != "IUPAC"]
        tax[classes] = tax[classes].map(lambda x: min(1, x))

        # Convert IUPAC names to SMILES and remove those that cannot be converted
        smiles = SMILESStorage(root)
        tax["SMILES"] = tax["IUPAC"].apply(lambda x: smiles.query(x) or "")
        smiles.close()
        tax = tax[tax["SMILES"] != ""]
        tax = tax[tax["SMILES"].notna()]

        # Remove all classes that have less than 10 annotations and glycans with no annotations
        cols = [x for x in tax.columns if x not in {"IUPAC", "SMILES"} and tax[x].values.sum() >= MIN_FREQUENCY]
        tax = tax[["IUPAC", "SMILES"] + cols]
        tax = tax[tax[cols].values.sum(axis=1) > 0.5].astype({c: int for c in cols})

        # Extract the stratification for the datasail splits
        strat = {}
        for _, row in tax.iterrows():
            strat[row["IUPAC"]] = list(row[cols].values)

        # Compute the datasail splits
        e_splits, _, _ = datasail(
            techniques=["I1e"],
            splits=[7, 2, 1],
            names=["train", "val", "test"],
            e_type="O",
            e_data={x: x for x in tax["IUPAC"].tolist()},
            e_strat=strat,
            epsilon=0.2,
            delta=0.2,
        )
        tax["split"] = tax["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))
        tax.to_csv(p, sep="\t", index=False)
    return p


def get_tissue(root: Path) -> Path:
    """
    Load the tissue data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.
    
    Returns:
        The filepath of the processed tissue data.
    """
    if not (p := (root / Path("tissue.tsv"))).exists():
        # Read in the UBERON ontology and the glycan tissue data
        UBERON = from_file("uberon.owl").graph
        df = build_custom_df(df_glycan, "df_tissue")[["glycan", "tissue_id"]]

        # Remove all unknown tissue ids
        df = df[df["tissue_id"].apply(lambda x: x in UBERON.nodes)]
        
        # Canonicalize the IUPAC names
        df["IUPAC"] = df["glycan"].apply(standardize_iupac)
        df = df[df["IUPAC"].notna()]

        # Bubble up tissue ids to our roots of the UBERON ontology
        df["uberon"] = df["tissue_id"].apply(lambda x: list(set(nx.ancestors(UBERON, x)).intersection(UBERON_ROOTS)))

        # Compute onehot encodings for the UBERON IDs
        vecs = np.stack([to_oh(names) for names in df["uberon"].values])
        tissue = pd.concat([df[["IUPAC"]], pd.DataFrame(vecs, columns=UBERON_ROOTS, dtype=int)], axis=1)
        
        # Group by IUPAC and sum the one-hot encodings
        tissue = tissue.groupby("IUPAC").agg("sum").reset_index()

        # Convert IUPAC names to SMILES and remove those that cannot be converted
        smiles = SMILESStorage(root)
        tissue["SMILES"] = tissue["IUPAC"].apply(lambda x: smiles.query(x))
        smiles.close()
        tissue = tissue[tissue["SMILES"] != ""]
        tissue = tissue[tissue["SMILES"].notna()]

        # Remove all classes that have less than 15 annotations
        keep = ["IUPAC", "SMILES"] + [x for x in tissue.columns if ":" in x and tissue[x].values.sum() >= MIN_FREQUENCY]
        tissue = tissue[keep]
        
        # Remove all glycans that do not have any tissue annotations
        class_names = [c for c in tissue.columns if ":" in c]
        tissue = tissue[tissue[class_names].values.sum(axis=1) > 0.5].astype({c: int for c in class_names})
        tissue[class_names] = tissue[class_names].applymap(lambda x: min(x, 1))
        
        # Extract the tissue stratification for the datasail splits
        strat = {}
        for _, row in tissue.iterrows():
            strat[row["IUPAC"]] = row[class_names].values.tolist()
        
        # Convert the IUPAC names to SMILES for the datasail splits
        e_splits, _, _ = datasail(
            techniques=["I1e"],
            splits=[7, 2, 1],
            names=["train", "val", "test"],
            e_type="O",
            e_data={x: x for x in tissue["IUPAC"].tolist()},
            e_strat=strat,
            epsilon=0.3,
            delta=0.3,
        )
        tissue["split"] = tissue["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))

        tissue.to_csv(p, sep="\t", index=False)
    return p


def get_immunogenicity(root: Path | str) -> Path:
    """
    Download immunogenicity data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.

    Returns:
        The filepath of the processed immunogenicity data.
    """
    root = Path(root)
    if not (p := (root / "immunogenicity.tsv")).exists():
        # Download the data
        urllib.request.urlretrieve("https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_immunogenicity.csv", p.with_suffix(".csv"))

        # Process the data and remove unnecessary columns
        df = pd.read_csv(p.with_suffix(".csv"))[["glycan", "immunogenicity"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        # One-hot encode the individual classes and collate them for glycans that are the same
        classes = {n: i for i, n in enumerate(df["immunogenicity"].unique())}
        df["label"] = df["immunogenicity"].map(classes)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])

        df.drop("immunogenicity", axis=1, inplace=True)
        df.to_csv(p, sep="\t", index=False)
        with open(root / "immunogenicity_classes.tsv", "w") as f:
            for n, i in classes.items():
                print(n, i, sep="\t", file=f)
    return p


def get_glycosylation(root: Path | str) -> Path:
    """
    Download glycosylation data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.

    Returns:
        The filepath of the processed glycosylation data.
    """
    root = Path(root)
    if not (p := root / "glycosylation.tsv").exists():
        # Read in the glycosylation data
        link = df_glycan[["glycan", "glycan_type"]]

        # Standardize the IUPAC names
        link["IUPAC"] = link["glycan"].apply(standardize_iupac)
        link = link[link["IUPAC"].notna()]

        # Drop duplicate IUPAC names
        link.drop_duplicates("IUPAC", inplace=True)

        # Translate IUPAC names to SMILES and remove those that cannot be converted
        smiles = SMILESStorage(root)
        link["SMILES"] = link["IUPAC"].apply(lambda x: smiles.query(x))
        smiles.close()
        link = link[link["SMILES"] != ""]
        link = link[link["SMILES"].notna()]
        # Remove classes with less than 15 annotations
        keep = set([k for k, v in dict(link["glycan_type"].value_counts()).items() if v >= MIN_FREQUENCY])
        link = link[link["glycan_type"].isin(keep)]

        # Compute datasplit with datasail
        e_splits, _, _ = datasail(
            techniques=["I1e"],
            splits=[7, 2, 1],
            names=["train", "val", "test"],
            e_type="O",
            e_data={x: x for x in link["IUPAC"].tolist()},
            e_strat=dict(link[["IUPAC", "glycan_type"]].values),
            epsilon=0.2,
            delta=0.2,
        )
        link["split"] = link["IUPAC"].map(lambda x: e_splits["I1e"][0].get(x, None))
        
        # Save the processed data
        link.to_csv(p.parent / "glycosylation_raw.tsv", sep="\t", index=False)

        class_map = {label: i for i, label in enumerate(link["glycan_type"].unique())}
        with open(p.parent / "glycosylation_classes.tsv", "w") as f:
            for label, i in class_map.items():
                print(label, i, sep="\t", file=f)
        link["label"] = link["glycan_type"].map(class_map)
        link = link[["IUPAC", "SMILES", "label", "split"]]
        link.to_csv(p, sep="\t", index=False)

    return p


def get_spectrum(root: Path | str) -> Path:
    """
    Download spectrum data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.

    Returns:
        The filepath of the processed spectrum data.
    """
    suffix = "_small"
    root = Path(root)
    if not (p := root / f"spectrum{suffix}.tsv").exists():
        # df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "Gothenburg" / "GIFFLAR" / "spectra_data" / f"spectrum_2048{suffix}.tsv", sep="\t")
        # df = pd.read_csv(Path("/") / "scratch" / "spectra_data" / f"spectrum_2048{suffix}.tsv", sep="\t")
        df = pd.read_csv(Path("/") / "scratch" / "chair_kalinina" / "s8rojoer" / "GIFFLAR" / "data_new_256" / f"spectrum_2048{suffix}.tsv", sep="\t")
        df.to_csv(p, sep="\t", index=False)
    return p


def get_dataset(data_config: dict, root: Path | str) -> dict:
    """
    Get the dataset based on the configuration.

    Args:
        data_config: The configuration of the dataset.
        root: The root directory to save the data to.

    Returns:
        The configuration of the dataset with the filepath added and made sure the dataset is preprocessed
    """
    Path(root).mkdir(exist_ok=True, parents=True)
    name_fracs = data_config["name"].split("_")
    match name_fracs[0]:
        case "Taxonomy":
            path = get_taxonomic_level(root, name_fracs[1])
        case "Tissue":
            path = get_tissue(root)
        case "Immunogenicity":
            path = get_immunogenicity(root)
        case "Glycosylation":
            path = get_glycosylation(root)
        case "Spectrum":
            path = get_spectrum(root)
        case "class-1" | "class-n" | "multilabel" | "reg-1" | "reg-n":  # Used for testing
            base = Path("dummy_data")
            if not base.is_dir():
                base = "tests" / base
            path = base / f"{name_fracs[0].replace('-', '_')}.csv"
        case _:  # Unknown dataset
            raise ValueError(f"Unknown dataset {data_config['name']}.")
    data_config["filepath"] = path
    return data_config
