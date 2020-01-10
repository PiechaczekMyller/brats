import typing

import torch
import numpy as np
import medpy.metric.binary as mp

import brats.utils as utils

CHANNELS_DIM = 1
ONE_CLASS = 1
FIRST_CLASS = 0


class DiceScoreOneCLass:
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
        assert prediction.shape[CHANNELS_DIM] == ONE_CLASS and \
               target.shape[CHANNELS_DIM] == ONE_CLASS
        all_but_batch_dims = list(range(1, target.dim()))
        intersection = utils.calculate_intersection(prediction, target,
                                                    dim=all_but_batch_dims)
        union = utils.calculate_union(prediction, target,
                                      dim=all_but_batch_dims)
        score = (2. * intersection) / (union + self.epsilon)
        return torch.mean(score)


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
        assert utils.is_binary(prediction), "Predictions must be binary"
        assert utils.is_binary(target), "Target must be binary"
        all_but_batch_dims = list(range(1, target.dim()))
        true_positives = utils.calculate_intersection(prediction, target,
                                                      dim=all_but_batch_dims)
        false_negatives = utils.calculate_false_negatives(prediction, target,
                                                          dim=all_but_batch_dims)
        recall = true_positives / (
                true_positives + false_negatives + self.epsilon)
        return torch.mean(recall)


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
        assert utils.is_binary(prediction), "Predictions must be binary"
        assert utils.is_binary(target), "Target must be binary"
        all_but_batch_dims = list(range(1, target.dim()))
        true_positives = utils.calculate_intersection(prediction, target,
                                                      dim=all_but_batch_dims)
        false_positive = utils.calculate_false_positives(prediction, target,
                                                         dim=all_but_batch_dims)
        precision = true_positives / (
                    true_positives + false_positive + self.epsilon)
        return torch.mean(precision)


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
        self.precision_score = PrecisionScore()
        self.recall_score = RecallScore()

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
        precision = self.precision_score(prediction, target)
        recall = self.recall_score(prediction, target)
        f_score = (1 + self.beta ** 2) * ((precision * recall) / (
                self.beta ** 2 * precision + recall + self.epsilon))
        return torch.mean(f_score)


class HausdorffDistance95:
    def __init__(self, merge_operation=torch.max):
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
        assert utils.is_binary(prediction), "Predictions must be binary"
        assert utils.is_binary(target), "Target must be binary"
        prediction = prediction.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        volumes_count, _, slices_count, _, _ = prediction.shape
        results = np.zeros((volumes_count, slices_count))
        for volume_idx in range(volumes_count):
            for slice_idx in range(slices_count):
                prediction_slice = prediction[
                    volume_idx, FIRST_CLASS, slice_idx, ...]
                target_slice = target[volume_idx, FIRST_CLASS, slice_idx, ...]
                results[volume_idx, slice_idx] = mp.hd95(prediction_slice,
                                                         target_slice)
        return self.merge_operation(torch.from_numpy(results))
