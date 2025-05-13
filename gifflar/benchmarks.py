import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species as taxonomy
from tqdm import tqdm

from gifflar.utils import iupac2smiles


def get_taxonomy(root: Path | str) -> pd.DataFrame:
    """
    Download full taxonomy data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.

    Returns:
        The processed taxonomy data.
    """
    if not (p := (root / Path("taxonomy.tsv"))).exists():
        mask = []
        # convert to IUPAC to SMILES and build a mask to remove not-convertable molecules.
        for i in tqdm(taxonomy["glycan"]):
            smiles = iupac2smiles(i)
            mask.append(smiles is not None)
        tax = taxonomy[mask]
        tax.to_csv(p, sep="\t", index=False)
    return pd.read_csv(p, sep="\t")


def get_taxonomic_level(
        root: Path | str,
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
        # Chop to taxonomic level of interest and remove invalid rows
        tax = get_taxonomy(root)[["glycan", level]]
        tax.rename(columns={"glycan": "IUPAC"}, inplace=True)
        tax[tax[level] == "undetermined"] = np.nan
        tax.dropna(inplace=True)

        # One-hot encode the individual classes and collate them for glycans that are the same
        tax = pd.concat([tax["IUPAC"], pd.get_dummies(tax[level])], axis=1)
        tax = tax.groupby('IUPAC').agg("sum").reset_index()

        # Chop prediction values to 0 and 1
        classes = [x for x in tax.columns if x != "IUPAC"]
        tax[classes] = tax[classes].applymap(lambda x: min(1, x))

        tax["split"] = np.random.choice(["train", "val", "test"], tax.shape[0], p=[0.7, 0.2, 0.1])
        mask = ((tax[tax["split"] == "train"][classes].sum() == 0) |
                (tax[tax["split"] == "val"][classes].sum() == 0) |
                (tax[tax["split"] == "test"][classes].sum() == 0))
        tax.drop(columns=np.array(classes)[mask], inplace=True)
        classes = [x for x in tax.columns if x not in {"IUPAC", "split"}]
        tax = tax[tax[classes].sum(axis=1) > 0]
        tax.to_csv(p, sep="\t", index=False)
    return p


def get_tissue(root: Path | str) -> Path:
    """
    Load the tissue data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.
    
    Returns:
        The filepath of the processed tissue data.
    """
    if not (p := (root / Path("tissue.tsv"))).exists():
        # Process the data and remove unnecessary columns
        df = pd.read_csv("datasets/tissue_multilabel_df.csv")
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])
        df.to_csv(p, sep="\t", index=False)
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
        urllib.request.urlretrieve("https://torchglycan.s3.us-east-2.amazonaws.com/downstream/glycan_properties.csv", p.with_suffix(".csv"))
        df = pd.read_csv(p.with_suffix(".csv"))[["glycan", "link"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        classes = {n: i for i, n in enumerate(df["link"].unique())}
        df["label"] = df["link"].map(classes)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])

        df.drop("link", axis=1, inplace=True)
        df.to_csv(p, sep="\t", index=False)
        with open(root / "glycosylation_classes.tsv", "w") as f:
            for n, i in classes.items():
                print(n, i, sep="\t", file=f)
    return p


def get_spectrum(root: Path | str) -> Path:
    """
    Download spectrum data, process it, and save it as a tsv file.

    Args:
        root: The root directory to save the data to.

    Returns:
        The filepath of the processed spectrum data.
    """
    root = Path(root)
    #if not (p := root / "spectrum.tsv").exists():
    p = root / "spectrum.tsv"
    df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "Gothenburg" / "GIFFLAR" / "spectrum_pred_512_small.tsv", sep="\t")
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
