import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Literal, Any, Generator

import yaml
from glycowork.glycan_data.loader import lib
from glyles import convert
from rdkit import Chem
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.svm import LinearSVR, SVC, SVR
from torchmetrics import CosineSimilarity, KLDivergence, MetricCollection, Accuracy, AUROC, MatthewsCorrCoef, MeanAbsoluteError, MeanSquaredError, \
    R2Score

from gifflar.metrics import Sensitivity

# MASK, unknown, values...
atom_map = {6: 2, 7: 3, 8: 4, 15: 5, 16: 6}
# bond_map = {Chem.BondType.SINGLE: 0, Chem.BondType.AROMATIC: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}
bond_map = {Chem.BondDir.BEGINDASH: 2, Chem.BondDir.BEGINWEDGE: 3, Chem.BondDir.NONE: 4}
lib_map = {n: i + 2 for i, n in enumerate(lib)}
mono_map = {n: i + 1 for i, n in enumerate([
    'Glc', 'Man', 'Gal', 'Gul', 'Alt', 'All', 'Tal', 'Ido', 'Qui', 'Rha', 'Fuc', 'Oli', 'Tyv', 'Abe', 'Par', 'Dig',
    'Col', 'Ara', 'Lyx', 'Xyl', 'Rib', 'Kdn', 'Neu', 'Sia', 'Pse', 'Leg', 'Aci', 'Bac', 'Kdo', 'Dha', 'Mur', 'Api',
    'Fru', 'Tag', 'Sor', 'Psi', 'Ery', 'Thre', 'Rul', 'Xul', 'Unk', 'Ace', 'Aco', 'Asc', 'Fus', 'Ins', 'Ko', 'Pau',
    'Per', 'Sed', 'Sug', 'Vio', 'Xlu', 'Yer', 'Erwiniose'
])}
mods_map = {n: i + 1 for i, n in enumerate([
    'Ac', 'DD', 'DL', 'LD', 'LL', 'en', 'A', 'N', 'F', 'I', 'S', 'P', 'NAc', 'ol', 'Me'
])}


def get_number_of_classes(cell: str) -> int:
    """Get the number of classes for a given cell."""
    match (cell):
        case "atoms":
            return len(atom_map) + 1
        case "bonds":
            return len(bond_map) + 1
        case "monosacch":
            return len(lib_map) + 1
        case _:
            return 0


def read_yaml_config(filename: str | Path) -> dict:
    """Read in yaml config for training."""
    with open(filename, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


def merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dictionaries a and b."""
    out = a
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            out[key] = merge_dicts(a[key], b[key])
        else:
            out[key] = b[key]
    return out


def unfold_config(config: dict) -> Generator[dict, None, None]:
    """
    Unfold the configuration by expanding multiple model and dataset settings into individual configs.

    Params:
        config: The configuration to unfold.

    Yields:
        The unfolded configuration.
    """
    if isinstance(config["datasets"], dict):
        datasets = [config["datasets"]]
    else:
        datasets = config["datasets"]
    del config["datasets"]

    if isinstance(config["model"], dict):
        models = [config["model"]]
    else:
        models = config["model"]
    del config["model"]
    if models is None:
        models = [{}]

    for dataset in datasets:
        for model in models:
            tmp_config = copy.deepcopy(config)
            tmp_config["dataset"] = dataset
            if "label" in tmp_config["dataset"] and not isinstance(tmp_config["dataset"]["label"], list):
                tmp_config["dataset"]["label"] = [tmp_config["dataset"]["label"]]
            tmp_config["model"] = model
            yield tmp_config


def hash_dict(input_dict: dict, n_chars: int = 8) -> str:
    """
    Generate a hash of a dictionary.

    Params:
        input_dict: The dictionary to hash.
        n_chars: The number of characters to include in the hash.

    Returns:
        The hash of the dictionary.
    """
    # Convert the dictionary to a JSON string
    dict_str = json.dumps(input_dict, sort_keys=True)

    # Generate a SHA-256 hash of the string
    hash_obj = hashlib.sha256(dict_str.encode())

    # Get the first 8 characters of the hexadecimal digest
    hash_str = hash_obj.hexdigest()[:n_chars]

    return hash_str


def iupac2smiles(iupac: str) -> Optional[str]:
    """
    Convert IUPAC-condensed representations to SMILES strings (or None if the smiles string cannot be valid).

    Args:
        iupac: The IUPAC-condensed representation of the glycan.

    Returns:
        The SMILES string representation of the glycan.
    """
    if any([x in iupac for x in ["?", "{", "}"]]):
        return None
    try:
        smiles = convert(iupac)[0][1]
        if len(smiles) < 10:
            return None
        return smiles
    except:
        return None


def get_mods_list(node: dict[str, Any]) -> list[int]:
    """Compute the one-hot encoding of modifications of a monosaccharides information provided in node."""
    ids = [0] * 16
    for mod in node["recipe"]:
        if mod != node["name"]:
            ids[mods_map.get(re.sub(r'[^A-Za-z]', '', mod[0]), 0)] = 1
    return ids


def get_sl_model(
        name: Literal["rf", "svm", "xgb"],
        task: Literal["regression", "classification", "multilabel", "spectrum"],
        n_outputs: int,
        **kwargs: Any,
) -> BaseEstimator:
    """
    Retrieve a statistical learning model and initialize it based on the dataset.

    Args:
        name: The name of the statistical learning model
        task: The type of the prediction task
        n_outputs: Number of predictions to be made
        **kwargs: Additional arguments to the model

    Returns:
        Ready-to-fit statistical learning model
    """
    if task == "spectrum":
        task = "regression"
    model_args = {k: v for k, v in kwargs["model"].items() if k not in {"name", "featurization"}}
    match name, task, n_outputs:
        case "rf", "regression", _:
            return RandomForestRegressor(**model_args)
        case "rf", _, _:
            return RandomForestClassifier(**model_args)
        case "svm", "regression", 1:
            return LinearSVR(**model_args)
        case "svm", "regression", _:
            if "random_state" in model_args:
                del model_args["random_state"]
            return MultiOutputRegressor(SVR(kernel="linear", **model_args))
        case "svm", "classification", _:
            return SVC(kernel="linear", probability=True, **model_args)
        case "svm", "multilabel", _:
            return MultiOutputClassifier(SVC(kernel="linear", probability=True, **model_args))
        case "xgb", "regression", 1:
            return GradientBoostingRegressor(**model_args)
        case "xgb", "regression", _:
            return MultiOutputRegressor(GradientBoostingRegressor(**model_args))
        case "xgb", "classification", 1:
            return GradientBoostingClassifier(**model_args)
        case "xgb", "classification", _:
            return MultiOutputClassifier(GradientBoostingClassifier(**model_args))
        case "xgb", "multilabel", _:
            return MultiOutputClassifier(GradientBoostingClassifier(**model_args))
        case _:
            raise NotImplementedError(f"The combination of (name, task, n_outputs) as ({name} {task}, {n_outputs}) has "
                                      f"not been considered.")


def get_metrics(
        task: Literal["regression", "classification", "multilabel", "spectrum"],
        n_outputs: int,
        prefix: str = "",
) -> dict[str, MetricCollection]:
    """
    Collect the metrics to monitor a models learning progress. For regression tasks, we monitor
      - the MSE,
      - the MAE, and
      - the R2Score
    For everything else (single- and multilabel classification) we monitor
      - the Accuracy,
      - the AUROC,
      - the MCC, and
      - the Sensitivity, i.e. the TPR

    Args:
        task: The type of the prediction task
        n_outputs: Number of predictions to be made

    Returns:
        A dictionary mapping split names to torchmetrics.MetricCollection
    """
    if task == "spectrum":
        m = MetricCollection([
            CosineSimilarity(),
            KLDivergence(),
        ])
    elif task == "regression":
        m = MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError(),
            R2Score(num_outputs=n_outputs),
        ])
    else:
        # Based on the number of outputs, the metrics need different arguments
        if n_outputs == 1:
            metric_args = {"task": "binary"}
        elif task != "multilabel":
            metric_args = {"task": "multiclass", "num_classes": n_outputs}
        else:
            metric_args = {"task": "multilabel", "num_labels": n_outputs}
        m = MetricCollection([
            Accuracy(**metric_args),
            AUROC(**metric_args),
            MatthewsCorrCoef(**metric_args),
            Sensitivity(**metric_args),
        ])
    return {"train": m.clone(prefix="train/" + prefix), "val": m.clone(prefix="val/" + prefix),
            "test": m.clone(prefix="test/" + prefix)}
