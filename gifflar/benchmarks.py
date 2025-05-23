import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from glycowork.glycan_data.loader import df_species, df_glycan, build_custom_df
from tqdm import tqdm

from gifflar.utils import iupac2smiles


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
        tax = df_species[["glycan", level]]
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
        df = build_custom_df(df_glycan, "df_tissue")[["glycan", "tissue_sample"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)

        # One-hot encode the individual classes and collate them for glycans that are the same
        df = pd.concat([df["IUPAC"], pd.get_dummies(df["tissue_sample"])], axis=1)
        df = df.groupby('IUPAC').agg("sum").reset_index()

        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])
        df.to_csv(p, sep="\t", index=False)
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
        df = df_glycan[["glycan", "glycan_type"]]
        df.rename(columns={"glycan": "IUPAC"}, inplace=True)
        df.dropna(inplace=True)

        classes = {n: i for i, n in enumerate(df["glycan_type"].unique())}
        df["label"] = df["glycan_type"].map(classes)
        df["split"] = np.random.choice(["train", "val", "test"], df.shape[0], p=[0.7, 0.2, 0.1])

        df.drop("glycan_type", axis=1, inplace=True)
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
    # root = Path(root)
    # if not (p := root / "spectrum.tsv").exists():
    #     p = root / "spectrum.tsv"
    #     df = pd.read_csv(Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "Gothenburg" / "GIFFLAR" / "spectrum_pred_2048.tsv", sep="\t")
    #     df.to_csv(p, sep="\t", index=False)
    return Path("/") / "scratch" / "SCRATCH_SAS" / "roman" / "Gothenburg" / "GIFFLAR" / "spectrum_pred_2048.tsv"


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
        case "Glycosylation" | "Linkage":
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
