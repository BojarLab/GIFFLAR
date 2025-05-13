import pickle
from pathlib import Path
from typing import Union, Optional, Callable, Any
import os
import shutil

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, OnDiskDataset, Database, SQLiteDatabase
from torch_geometric.data.data import BaseData
from torch_geometric.data.database import Schema
from tqdm import tqdm

from gifflar.data.utils import GlycanStorage


PROCESS_CHUNK_SIZE = 10_000

class GlycanOnDiskDataset(OnDiskDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            schema: Schema = object,
            path_idx: int = 0,
            log: bool = True,
            force_reload: bool = False,
    ) -> None:
        self.backend = "sqlite"  # "sqlite" or "rocksdb"
        self.schema = schema
        self.path_idx = path_idx

        self._db: Optional[Database] = None
        self._numel: Optional[int] = None
        super(OnDiskDataset, self).__init__(root, transform, pre_transform, pre_filter, log=log, force_reload=force_reload)
        self.dataset_args = torch.load(Path(self.processed_paths[path_idx]).with_suffix(".pth"))

    def _process(self):
        if self.force_reload:
            print("Remove root", self.root)
            shutil.rmtree(self.root, ignore_errors=True)
            self.force_reload = False
        super(OnDiskDataset, self)._process()

    def get_db(self, path_idx):
        kwargs = {}
        cls = self.BACKENDS[self.backend]
        if issubclass(cls, SQLiteDatabase):
            kwargs['name'] = self.__class__.__name__

        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[path_idx]
        db = cls(path=path, schema=self.schema, **kwargs)
        return db

    @property
    def db(self) -> Database:
        r"""Returns the underlying :class:`Database`."""
        if self._db is None:
            self._db = self.get_db(self.path_idx)
            self._numel = len(self._db)
        return self._db

    def get(self, idx: int) -> BaseData:
        d = super().get(idx)
        if self.transform is not None:
            d = self.transform(d)
        return d

    def __getitem__(self, idx: int):
        return self.get(idx)

    def __getitems__(self, indices):
        return [self.get(idx) for idx in indices]

    def process_(self, data: list[HeteroData], path_idx: int = 0, final: bool = True) -> None:
        """

        """
        if len(data) != 0:
            print("Processing", len(data), "entries")
            self.db.multi_insert(range(self._numel, self._numel + len(data)), data, batch_size=None)
            self._numel += len(data)
        if final:
            self.db.close()
            self._db = None
            print("Saving dataset arguments to", Path(self.processed_paths[path_idx]).with_suffix(".pth"))
            torch.save(self.dataset_args, Path(self.processed_paths[path_idx]).with_suffix(".pth"))
    
    def serialize(self, data: BaseData) -> Any:
        r"""Serializes the :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object into the expected DB
        schema.
        """
        return data

    def deserialize(self, data: Any) -> BaseData:
        r"""Deserializes the DB entry into a
        :class:`~torch_geometric.data.Data` or
        :class:`~torch_geometric.data.HeteroData` object.
        """
        return data


class GlycanDataset(GlycanOnDiskDataset):
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
        super().__init__(root=str(Path(root) / f"{self.filename.stem}_{hash_code}"), schema=schema,
                         transform=transform, pre_transform=pre_transform, path_idx=path_idx, force_reload=force_reload)

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

        super(GlycanDataset, self).process_(data, path_idx, final)


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
        return [self.filename.stem + ".pt"]

    def process(self) -> None:
        """Process the data and store it."""
        data = []
        gs = GlycanStorage(Path(self.root).parent)
        with open(self.filename, "r") as glycans:
            for i, line in enumerate(glycans.readlines()):
                if len(data) % PROCESS_CHUNK_SIZE == 0:
                    self.process_(data, final=False)
                    del data
                    data = []
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
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        self.split = split
        super().__init__(root=root, filename=filename, hash_code=hash_code, schema=schema, transform=transform,
                         pre_transform=pre_transform, path_idx=self.splits[split], force_reload=force_reload, **dataset_args)

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        """Return the list of processed file names."""
        return [split + ".pt" for split in self.splits.keys()]

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
            self.dataset_args["label"] = [x for x in df.columns if x not in {"IUPAC", "split", "red_mz"}]

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
            if len(data) % PROCESS_CHUNK_SIZE == 0:
                self.process_(data, path_idx=self.splits[self.split], final=False)
                del data
                data = []
            if row["split"] != self.split:
                continue
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
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


class LGIDataset(DownstreamGDs):
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
            if len(data) % PROCESS_CHUNK_SIZE == 0:
                self.process_(data, path_idx=self.splits[self.split], final=False)
                del data
                data = []
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
        inter = pd.read_csv(self.filename, sep=sep)
        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (_, row) in tqdm(enumerate(inter.iterrows())):
            if len(data) % PROCESS_CHUNK_SIZE == 0:
                self.process_(data, path_idx=self.splits[self.split], final=False)
                del data
                data = []
            split = getattr(row, "split", "train")
            if split != self.split:
                continue
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
            d["aa_seq"] = row["seq"]
            d["y"] = torch.tensor([getattr(row, "y", 0)])
            d["ID"] = i
            data.append(d)
        
        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)


class ContrastiveLGIDataset(LGIDataset):
    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            split: str,
            hash_code: str,
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
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **dataset_args: Additional arguments to pass to the dataset
        """
        super().__init__(root=root, filename=filename, split=split, hash_code=hash_code, schema=(object, object), transform=transform,
                         pre_transform=pre_transform, force_reload=force_reload, **dataset_args)
    def _inter_process(self, data: list[HeteroData], path_idx: int) -> None:
        db = self.get_db(path_idx)
        if len(data) != 0:
            self.db.multi_insert(range(len(data)), data, batch_size=None)
        db.close()
        torch.save(self.dataset_args, Path(self.processed_paths[path_idx]).with_suffix(".pth"))


    def process_pkl(self) -> None:
        with open(self.filename, "rb") as f:
            lgis = pickle.load(f)

        # Load the glycan storage to speed up the preprocessing
        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (lectin, glycan, glycan_val, decoy, decoy_val, split) in tqdm(enumerate(lgis)):
            if len(data) % PROCESS_CHUNK_SIZE == 0:
                self.process_(data, path_idx=self.splits[self.split], final=False)
                del data
                data = []
            if split != self.split:
                continue
            try:
                d = gs.query(glycan)
                if d is None:
                    continue
                d["aa_seq"] = lectin
                d["y"] = torch.tensor([glycan_val])
                d["ID"] = i

                decoy = gs.query(decoy)
                decoy["y"] = torch.tensor([decoy_val])
                decoy["ID"] = i

                data.append((d, decoy))
            except Exception as e:
                print(e)
                continue

        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)

    def process_csv(self, sep):
        inter = pd.read_csv(self.filename, sep=sep)
        gs = GlycanStorage(Path(self.root).parent)
        data = []
        for i, (_, row) in tqdm(enumerate(inter.iterrows())):
            if len(data) % PROCESS_CHUNK_SIZE == 0:
                self.process_(data, path_idx=self.splits[self.split], final=False)
                del data
                data = []
            split = getattr(row, "split", "train")
            if split != self.split:
                continue
            d = gs.query(row["IUPAC"])
            if d is None:
                continue
            d["aa_seq"] = row["seq"]
            d["y"] = torch.tensor([getattr(row, "y", 0)])
            d["ID"] = i
            decoy = gs.query(getattr(row, "decoy", None))
            data.append((d, decoy))
        
        gs.close()
        self.process_(data, path_idx=self.splits[self.split], final=True)
