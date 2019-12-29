import typing

import torch
import numpy as np
import medpy.metric.binary as mp

import brats.utils as utils

CHANNELS_DIM = 1
ONE_CLASS = 1
FIRST_CLASS = 0


def dice_score_one_class(prediction: torch.Tensor, target: torch.Tensor,
                         epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute an average DICE score across a batch of volumes, containing only
    prediction for one class
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: DICE score for each element in the batch
    """
    assert prediction.shape[CHANNELS_DIM] == ONE_CLASS and \
           target.shape[CHANNELS_DIM] == ONE_CLASS
    all_but_batch_dims = list(range(1, target.dim()))
    intersection = utils.calculate_intersection(prediction, target,
                                                dim=all_but_batch_dims)
    union = utils.calculate_union(prediction, target, dim=all_but_batch_dims)
    score = (2. * intersection) / (union + epsilon)
    return score


def recall_score(prediction: torch.Tensor, target: torch.Tensor,
                 epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the recall score
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: Recall score for each element in the batch
    """
    assert utils.is_binary(prediction), "Predictions must be binary"
    assert utils.is_binary(target), "Target must be binary"
    all_but_batch_dims = list(range(1, target.dim()))
    true_positives = utils.calculate_intersection(prediction, target,
                                                  dim=all_but_batch_dims)
    false_negatives = utils.calculate_false_negatives(prediction, target,
                                                      dim=all_but_batch_dims)
    return true_positives / (true_positives + false_negatives + epsilon)


def precision_score(prediction: torch.Tensor, target: torch.Tensor,
                    epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the precision score
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: Precision score for each element in the batch
    """
    assert utils.is_binary(prediction), "Predictions must be binary"
    assert utils.is_binary(target), "Target must be binary"
    all_but_batch_dims = list(range(1, target.dim()))
    true_positives = utils.calculate_intersection(prediction, target,
                                                  dim=all_but_batch_dims)
    false_positive = utils.calculate_false_positives(prediction, target,
                                                     dim=all_but_batch_dims)
    return true_positives / (true_positives + false_positive + epsilon)


def f_score(prediction: torch.Tensor, target: torch.Tensor, beta: float,
            epsilon: float = 1e-6) -> torch.Tensor:
    """
    Compute the F score
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
        beta: Weight parameter between precision and recall
        epsilon: Smooth factor to prevent division by 0.
    Returns:
        torch.Tensor: F Score value for each element in the batch
    """
    precision = precision_score(prediction, target)
    recall = recall_score(prediction, target)
    return (1 + beta ** 2) * ((precision * recall) / (
            beta ** 2 * precision + recall + epsilon))


def hausdorff_distance95(prediction: torch.Tensor,
                         target: torch.Tensor) -> torch.Tensor:
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD)
    between the binary objects in two images.
    Heavily depends on medpy.metric.binary.hd95, so check it out for more
    details.
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
    return torch.from_numpy(results)


def hausdorff_distance95_mean_max(prediction: torch.Tensor,
                                  target: torch.Tensor) -> \
                                  typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Merges the hausdorff distances for each volume, using mean and max.
    Args:
        prediction: Network output.
            Dimensions - (Batch, Class, Depth, Height, Width)
        target: Target values.
            Dimensions - (Batch, Class, Depth, Height, Width)
    Returns:
        tuple: Tuple with max and average for each volume.
    """
    results_per_slice = hausdorff_distance95(prediction, target)
    return torch.mean(results_per_slice, dim=CHANNELS_DIM), \
           torch.max(results_per_slice, dim=CHANNELS_DIM)
