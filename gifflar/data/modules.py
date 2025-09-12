from pathlib import Path
from typing import Optional, Callable, Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

from gifflar.data.datasets import DownstreamGDs, PretrainGDs
from gifflar.data.datasets_lgi import LGIDataset, ContrastiveLGIDataset
from gifflar.data.hetero import hetero_collate, hetero_tuple_collate


class GlycanDataModule(LightningDataModule):
    """DataModule holding datasets for Glycan-specific training"""

    collate_fn = hetero_collate

    def __init__(
            self,
            batch_size: int = 128,
            num_workers: int = 0,
            **kwargs: Any
    ):
        """
        Initialize the DataModule with a given batch size.

        Args:
            batch_size: The batch size to use for training, validation, and testing
            num_workers: The number of CPUs to use for loading the data
            **kwargs: Additional arguments to pass to the Data
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the training data.

        Returns:
            DataLoader for the training data
        """
        batch_size = min(self.batch_size, len(self.train))
        return DataLoader(self.train, batch_size=batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=6, persistent_workers=True, 
                          drop_last=len(self.train) % batch_size == 1)

    def val_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the validation data.

        Returns:
            DataLoader for the validation data
        """
        return DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False,
                          collate_fn=self.collate_fn, num_workers=6, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        """
        Return the DataLoader for the test data.

        Returns:
            DataLoader for the test data
        """
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False,
                          collate_fn=self.collate_fn, num_workers=6, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        """
        Combines the train, val, and test datasets and return the DataLoader for that data.

        Returns:
            DataLoader for the combined data
        """
        if hasattr(self, "test"):
            predict = ConcatDataset([self.train, self.val, self.test])
        else:
            predict = ConcatDataset([self.train, self.val])
        return DataLoader(predict, batch_size=1, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=6, persistent_workers=True)


class PretrainGDM(GlycanDataModule):
    """DataModule for pretraining a model on glycan data."""

    def __init__(
            self,
            file_path: str | Path,
            hash_code: str,
            batch_size: int = 64,
            train_frac: float = 0.8,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            num_workers: int = 0,
            force_reload: bool = False,
            **kwargs: Any,
    ):
        """
        Initialize the DataModule with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            batch_size: The batch size to use for training, validation, and testing
            train_frac: The fraction of the data to use for training
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            **kwargs: Additional arguments to pass to the Pretrain
        """
        root = Path(file_path).parent
        super().__init__(batch_size, num_workers=num_workers)
        ds = PretrainGDs(root=root, filename=file_path, hash_code=hash_code, transform=transform,
                         pre_transform=pre_transform, force_reload=force_reload, **kwargs)
        self.train, self.val = torch.utils.data.dataset.random_split(ds, [train_frac, 1 - train_frac])


class DownstreamGDM(GlycanDataModule):
    """DataModule for downstream tasks on glycan data."""

    ds_class = DownstreamGDs

    def __init__(
            self,
            root: str | Path,
            filename: str | Path,
            hash_code: str,
            batch_size: int = 64,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            force_reload: bool = False,
            num_workers: int = 0,
            **dataset_args: dict[str, Any],
    ):
        """
        Initialize the DataModule with the given parameters.

        Args:
            root: The root directory to store the processed data
            filename: The filename of the data to process
            hash_code: The hash code to use for the processed data
            batch_size: The batch size to use for training, validation, and testing
            transform: The transform to apply to the data
            pre_transform: The pre-transform to apply to the data
            force_reload: Whether to force reload the data
            num_workers: The number of CPUs to use for loading the data
            **dataset_args: Additional arguments to pass to the DownstreamGDs
        """
        super().__init__(batch_size, num_workers=num_workers)
        self.train = self.ds_class(
            root=root, filename=filename, split="train", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, force_reload=force_reload, **dataset_args,
        )
        self.val = self.ds_class(
            root=root, filename=filename, split="val", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, force_reload=force_reload, **dataset_args,
        )
        self.test = self.ds_class(
            root=root, filename=filename, split="test", hash_code=hash_code, transform=transform,
            pre_transform=pre_transform, force_reload=force_reload, **dataset_args,
        )


class LGI_GDM(DownstreamGDM):
    ds_class = LGIDataset


class ConstrastiveGDM(DownstreamGDM):
    ds_class = ContrastiveLGIDataset
    collate_fn = hetero_tuple_collate
