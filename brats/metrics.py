import typing
from typing import Callable

import torch

import brats.functional as F

CHANNELS_DIM = 1
ONE_CLASS = 1
FIRST_CLASS = 0


class DiceScore:
    def __init__(self, epsilon: float = 1e-6):
        """
        Compute an average DICE score across a batch of volumes, for each class
        separately.
        Args:
            epsilon: Smooth factor for each element in the batch
        """
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """

        Args:
            prediction: Network output.
                Dimensions - (Batch, Class, Depth, Height, Width)
            target: Target values. The classes should be ont-hot encoded.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: DICE for each class score averaged
                across the whole batch
        """
        return F.dice(prediction, target, self.epsilon)


class RecallScore:
    def __init__(self, epsilon: float = 1e-6):
        """
        Compute the recall score
        Args:
            epsilon: Smooth factor to prevent division by 0.
        """
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Network output.
                Dimensions - (Batch, Class, Depth, Height, Width)
            target: Target values. The classes should be ont-hot encoded.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: Recall score for each class averaged
                across the whole batch
        """
        return F.recall(prediction, target, self.epsilon)


class PrecisionScore:
    def __init__(self, epsilon: float = 1e-6):
        """
        Compute the precision score
        Args:
            epsilon: Smooth factor to prevent division by 0.
        """
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Network output.
                Dimensions - (Batch, Class, Depth, Height, Width)
            target: Target values. The classes should be ont-hot encoded.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: Precision score for each class averaged
                across the whole batch
        """
        return F.precision(prediction, target, self.epsilon)


class FScore:
    def __init__(self, beta: float, epsilon: float = 1e-6):
        """
        Compute the F score
        Args:
            beta: Weight parameter between precision and recall
            epsilon: Smooth factor to prevent division by 0.
        """
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Network output.
                Dimensions - (Batch, Class, Depth, Height, Width)
            target: Target values. The classes should be ont-hot encoded.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: F Score for each class averaged
                across the whole batch
        """
        return F.f_score(prediction, target, self.beta, self.epsilon)


class HausdorffDistance95:
    def __init__(self, merge_operation: Callable[[torch.Tensor], float] = torch.max):
        """
        95th percentile of the Hausdorff Distance.

        Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD)
        between the binary objects in two images.
        Heavily depends on medpy.metric.binary.hd95, so check it out for more
        details.
        Args:
            merge_operation: Operation to merge results from all the slices in
                the batch into a single value. Should be a function that accepts
                two dimensional tensors (Batch, Depth).
                Good examples are: torch.mean, torch.max, torch.min.
        """
        self.merge_operation = merge_operation

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: Network output.
                Dimensions - (Batch, Class, Depth, Height, Width)
            target: Target values.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: Hausdorff distance for each slice.
                    Dimensions - (Batch, Depth)
            """
        return F.hausdorff95(prediction, target, self.merge_operation)


def metric_for_class(metric: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], class_id: int):
    """
    Wrapper for multiclass metrics that creates function that extracts metrics for single classes"
    :param metric: Function used to calculate metric
    :param class_id: Id of the class (from 0 to N)
    """

    def wrapper(prediction, target):
        return metric(prediction, target)[class_id]

    return wrapper
