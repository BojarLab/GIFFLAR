import pickle
from pathlib import Path
from typing import Union, Optional, Callable, Any
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, OnDiskDataset, Database, SQLiteDatabase, InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import Schema
from tqdm import tqdm

from gifflar.data.utils import GlycanStorage


class GlycanInMemoryDataset(InMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            log: bool = True,
            force_reload: bool = False,
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            path_idx: The index of the processed file name to use
            **dataset_args: Additional arguments to pass to the dataset
        """
        self.tmp_data_storage = []
        super().__init__(root, transform, pre_transform, pre_filter, log=log, force_reload=force_reload)
        self.data, self.dataset_args = torch.load(self.processed_paths[self.path_idx], weights_only=False)
        print(f"Loaded {len(self.data)} entries from {self.processed_paths[self.path_idx]} in memory.")

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.data.__len__()

    def __getitem__(self, item) -> Any:
        """Return the item at the given index."""
        return self.data[item] if self.transform is None else self.transform(self.data[item])

    def __getitems__(self, indices):
        data = [self.data[index] for index in indices]
        return data if self.transform is None else self.transform(data)

    def process_(self, data: list[HeteroData], path_idx: int = 0, final: bool = True) -> None:
        """
        Filter, process the data and store it at the given path index.

        Args:
            data: The data to process
            path_idx: The index of the processed file name to use
        """
        self.tmp_data_storage.extend(data)
        if final:
            torch.save((self.tmp_data_storage, self.dataset_args), self.processed_paths[path_idx])


class GlycanDataset(GlycanInMemoryDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            schema: Schema = object,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            path_idx: int = 0,
            force_reload: bool = False,
            **dataset_args: dict[str, Any],
    ):
        """
        Initialize the dataset with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            path_idx: The index of the processed file name to use
            **dataset_args: Additional arguments to pass to the dataset
        """
        self.filename = Path(filename)
        self.dataset_args = dataset_args
        self.path_idx = path_idx
        self.schema = schema
        super().__init__(root=str(Path(root) / f"{self.filename.stem}_{hash_code}"),
                         transform=transform, pre_transform=pre_transform, force_reload=force_reload)

    @property
    def processed_paths(self) -> list[str]:
        """Return the list of processed paths."""
        return [str(Path(self.root) / f) for f in self.processed_file_names]

    def process_(self, data: list[HeteroData], path_idx: int = 0, final: bool = True) -> None:
        """
        Filter, process the data and store it at the given path index.

        Args:
            data: The data to process
            path_idx: The index of the processed file name to use
        """
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        super().process_(data, path_idx, final)


class PretrainGDs(GlycanDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            schema: Schema = object,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
            **dataset_args: dict[str, Any],
    ):
        """
        Initialize the dataset for pre-training with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        super().__init__(root=root, filename=filename, hash_code=hash_code, schema=schema, transform=transform,
                         pre_transform=pre_transform, force_reload=force_reload, **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        """Return the list of processed file names"""
        return [self.filename.stem + ".pim"]

    def process(self) -> None:
        """Process the data and store it."""
        data = []
        gs = GlycanStorage(Path(self.root).parent)
        with open(self.filename, "r") as glycans:
            for i, line in enumerate(glycans.readlines()):
                d = gs.query(line.strip())
                d["ID"] = i
                data.append(d)
        gs.close()
        self.process_(data, final=True)


class DownstreamGDs(GlycanDataset):
    """Dataset for downstream tasks on glycan data."""
    splits = {"train": 0, "val": 1, "test": 2}

    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            split: str,
            hash_code: str,
            schema: Schema = object,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
            **dataset_args: dict[str, Any],
    ):
        """
        Initialize the dataset for downstream tasks with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            split: The split to use, e.g., train, val, test
            hash_code: The hash code to use for the processed data
            schema: The schema to use for the dataset
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            force_reload: Whether to force reload the dataset
            **dataset_args: Additional arguments to pass to the dataset
        """
        self.split = split
        print(split, self.splits[split])
        super().__init__(root=root, filename=filename, hash_code=hash_code, schema=schema, transform=transform,
                         pre_transform=pre_transform, path_idx=self.splits[split], force_reload=force_reload, **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        """Return the list of processed file names."""
        return [split + ".pim" for split in self.splits.keys()]

    def to_statistical_learning(self) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Convert the data to a format suitable for statistical learning (sklearn).

        Returns:
            The features, the labels, and the one-hot encoded labels
        """
        X, y, y_oh = [], [], []
        data_container = self.data if hasattr(self, "data") else self
        for d in data_container:
            X.append(d["fp"])
            y.append(d["y"])
            if hasattr(d, "y_oh"):
                y_oh.append(d["y_oh"])
        if isinstance(y[0], int):
            return np.vstack(X), np.array(y), np.vstack(y_oh) if len(y_oh) != 0 else None
        else:
            return np.vstack(X), np.concatenate(y), np.vstack(y_oh) if len(y_oh) != 0 else None

    def process(self) -> None:
        """Process the data and store it."""
        print("Start processing")
        df = pd.read_csv(self.filename, sep="\t" if self.filename.suffix.lower().endswith(".tsv") else ",")

        # If the label is not given, use all columns except IUPAC and split
        if "label" not in self.dataset_args:
            self.dataset_args["label"] = [x for x in df.columns if x not in {"IUPAC", "SMILES", "split", "red_mz"}]

        # Compute the number of classes
        if self.dataset_args["task"] != "classification":
            self.dataset_args["num_classes"] = len(self.dataset_args["label"])
        else:
            self.dataset_args["num_classes"] = int(max(df[self.dataset_args["label"]].values)) + 1
        if self.dataset_args["num_classes"] == 2:
            self.dataset_args["num_classes"] = 1

        # Load the glycan storage to speed up the preprocessing
        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (_, row) in tqdm(enumerate(df.iterrows())):
            if row["split"] != self.split:
                continue
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
            if "{" in row["IUPAC"] or "?" in row["IUPAC"]:
                print("SLIP:", row["IUPAC"])
                print(d)
            if self.dataset_args["task"] in {"regression", "spectrum"} or len(self.dataset_args["label"]) == 1:
                d["y"] = torch.tensor(list(row[self.dataset_args["label"]].values)).reshape(1, -1)
            elif len(self.dataset_args["label"]) > 1:
                d["y_oh"] = torch.tensor([int(x) for x in row[self.dataset_args["label"]]]).reshape(1, -1)
                if self.dataset_args["task"] != "multilabel":
                    d["y"] = d["y_oh"].argmax().item()
            d["ID"] = i
            if hasattr(row, "red_mz"):
                d["red_mz"] = row["red_mz"]
            data.append(d)

        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)


class IM_LGIDataset(DownstreamGDs):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            split: str,
            hash_code: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
            schema: Schema = object,
            **dataset_args: dict[str, Any],
    ):
        """
        Initialize the dataset for downstream tasks with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            split: The split to use, e.g., train, val, test
            hash_code: The hash code to use for the processed data
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        super().__init__(root=root, filename=filename, split=split, hash_code=hash_code, schema=schema, transform=transform,
                         pre_transform=pre_transform, force_reload=force_reload, **dataset_args)
    
    def process(self) -> None:
        """Process the data and store it."""
        if str(self.filename).endswith(".pkl"):
            self.process_pkl()
        elif str(self.filename)[-4:] in {".csv", ".tsv"}:
            self.process_csv(sep="," if str(self.filename)[-3] == "c" else "\t")

    def process_pkl(self) -> None:
        with open(self.filename, "rb") as f:
            inter, lectin_map, glycan_map = pickle.load(f)

        # Load the glycan storage to speed up the preprocessing
        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (lectin_id, glycan_id, value, split) in tqdm(enumerate(inter)):
            if split != self.split:
                continue
            d = gs.query(glycan_map[glycan_id])
            if d is None:
                continue
            d["aa_seq"] = lectin_map[lectin_id]
            d["y"] = torch.tensor([value])
            d["ID"] = i
            data.append(d)

        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)

    def process_csv(self, sep: str) -> None:
        DATA_BASE = self.filename.parent

        inter = pd.read_csv(self.filename, sep=sep)
        
        lectins = pd.read_csv(DATA_BASE / "lectins.csv")
        lectin_map = {x: y for x, y in lectins[["ID", "aa_seq"]].values}
        inter["aa_seq"] = inter["lectin"].map(lectin_map)

        glycans = pd.read_csv(DATA_BASE / "glycans.csv")
        glycan_map = {x: y for x, y in glycans[["ID", "IUPAC"]].values}
        inter["IUPAC"] = inter["glycan"].map(glycan_map)

        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (_, row) in tqdm(enumerate(inter.iterrows())):
            split = getattr(row, "split", "train")
            if split != self.split:
                continue
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
            d["aa_seq"] = row["aa_seq"]
            d["y"] = torch.tensor([getattr(row, "y", 0)], dtype=torch.float)
            d["ID"] = i
            data.append(d)
        
        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)
