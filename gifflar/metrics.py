from typing import Type, Literal, Optional, Any

import matchms
import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import BinaryConfusionMatrix, MulticlassConfusionMatrix, MultilabelConfusionMatrix
from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.utilities.enums import ClassificationTask


MAX = 3000
MIN = 49

class ModifiedCosineSimilarity(Metric):
    """Computes the modified cosine similarity between two sets of predictions.
    This metric is used to evaluate the similarity between predicted and true values in a batch.
    The modified cosine similarity is a measure of similarity between two spectra, which is commonly used in mass spectrometry.
    Args:
        tolerance: The tolerance value for the modified cosine similarity.
    """
    def __init__(self, num_bins: Optional[int] = None, tolerance: float = 0.2):
        super(ModifiedCosineSimilarity, self).__init__()
        self.mod_cos = matchms.similarity.ModifiedCosine(tolerance=tolerance)
        self.mz = None if num_bins is None else np.arange(MIN, MAX, (MAX - MIN) / num_bins)
        self.add_state("mod_cos_sim", default=[], dist_reduce_fx="cat")

    def update(self, pred: torch.Tensor, true: torch.Tensor, precursor_masses: list[float]) -> torch.Tensor:
        assert pred.shape == true.shape, "The predictions and true values must have the same shape"
        assert len(pred.shape) == 2, "The predictions must have dimensions (batch_size, num_predictions)"
        assert len(true.shape) == 2, "The true values must have dimensions (batch_size, num_predictions)"
        assert pred.shape[0] == len(precursor_masses), "The batch size the precursor masses must match those of predictions, and true values"
        
        if self.mz is None:
            self.mz = np.arange(MIN, MAX, (MAX - MIN) / pred.shape[1])
        self.mod_cos_sim += [float(self.mod_cos.pair(
            matchms.Spectrum(mz=self.mz, intensities=pred[i].cpu().numpy().astype(float), metadata={"precursor_mz": precursor_masses[i]}), 
            matchms.Spectrum(mz=self.mz, intensities=true[i].cpu().numpy().astype(float), metadata={"precursor_mz": precursor_masses[i]})
        )["score"]) for i in range(pred.shape[0])]
    
    def compute(self) -> torch.Tensor:
        """Computes the modified cosine similarity."""
        return torch.mean(torch.tensor(self.mod_cos_sim, dtype=float))


class BinarySensitivity(BinaryConfusionMatrix):
    """Computes sensitivity for binary classification tasks."""

    def compute(self) -> Tensor:
        """Computes the sensitivity."""
        confmat = super().compute().nan_to_num(0, 0, 0)
        denom = confmat[1, 1] + confmat[1, 0]
        return confmat[1, 1] / torch.maximum(denom, torch.full_like(denom, 1e-7, dtype=torch.float32))


class MulticlassSensitivity(MulticlassConfusionMatrix):
    """Computes sensitivity for multiclass classification tasks."""

    def compute(self) -> Tensor:
        """Computes the sensitivity as mean per-class sensitivity."""
        confmat = super().compute().nan_to_num(0, 0, 0)
        denom = confmat.sum(dim=1)
        return (confmat.diag() / torch.maximum(denom, torch.full_like(denom, 1e-7, dtype=torch.float32))).mean()


class MultilabelSensitivity(MultilabelConfusionMatrix):
    """Computes sensitivity for multilabel classification tasks."""

    def compute(self) -> Tensor:
        """Computes the sensitivity as mean per-class sensitivity"""
        confmat = super().compute().nan_to_num(0, 0, 0)
        denom = confmat[:, 1, :].sum(dim=1)
        return (self.confmat[:, 1, 1] / torch.maximum(denom, torch.full_like(denom, 1e-7, dtype=torch.float32))).mean()


class Sensitivity(_ClassificationTaskWrapper):
    def __new__(
            cls: Type["Sensitivity"],
            task: Literal["binary", "multiclass", "multilabel", "spectrum"],
            threshold: float = 0.5,
            num_classes: Optional[int] = None,
            num_labels: Optional[int] = None,
            normalize: Optional[Literal["true", "pred", "all", "none"]] = None,
            ignore_index: Optional[int] = None,
            validate_args: bool = True,
            **kwargs: Any,
    ) -> Metric:
        """
        Factory method to instantiate the appropriate sensitivity metric based on the task.

        Args:
            task: The classification task type.
            threshold: The threshold value for binary and multilabel classification tasks.
            num_classes: The number of classes for multiclass classification tasks.
            num_labels: The number of labels for multilabel classification tasks.
            normalize: Normalization mode for confusion matrix.
            ignore_index: The label index to ignore.
            validate_args: Whether to validate input args.
            **kwargs: Additional keyword arguments to pass to the metric.

        Returns:
            The sensitivity metric for the specified task.
        """
        # Initialize task metric
        task = ClassificationTask.from_str(task)
        kwargs.update({"normalize": normalize, "ignore_index": ignore_index, "validate_args": validate_args})
        if task == ClassificationTask.BINARY:
            return BinarySensitivity(threshold, **kwargs)
        if task == ClassificationTask.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`")
            return MulticlassSensitivity(num_classes, **kwargs)
        if task == ClassificationTask.MULTILABEL:
            if not isinstance(num_labels, int):
                raise ValueError(f"`num_labels` is expected to be `int` but `{type(num_labels)} was passed.`")
            return MultilabelSensitivity(num_labels, threshold, **kwargs)
        raise ValueError(f"Task {task} not supported!")
