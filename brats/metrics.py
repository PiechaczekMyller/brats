import torch

from brats.utils import calculate_intersection, calculate_union

CHANNELS_DIM = 1
ONE_CLASS = 1


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
    intersection = calculate_intersection(prediction, target,
                                          dim=all_but_batch_dims)
    union = calculate_union(prediction, target, dim=all_but_batch_dims)
    score = (2. * intersection) / (union + epsilon)
    return score
