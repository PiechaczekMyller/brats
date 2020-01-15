from typing import Callable

import torch

import brats.functional as F

CHANNELS_DIM = 1
ONE_CLASS = 1
FIRST_CLASS = 0


class DiceScoreOneClass:
    def __init__(self, epsilon: float = 1e-6):
        """
        Compute an average DICE score across a batch of volumes, containing only
        prediction for one class
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
            target: Target values.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: DICE score averaged across the whole batch
        """
        return F.dice_one_class(prediction, target, self.epsilon)


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
            target: Target values.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: Recall score averaged across the whole batch
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
            target: Target values.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: Precision score averaged across the whole batch
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
            target: Target values.
                Dimensions - (Batch, Class, Depth, Height, Width)
        Returns:
            torch.Tensor: F Score averaged across the whole batch
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
