import torch

ZERO_AND_ONE = torch.tensor([0., 1.])
CHANNELS_DIM = 1
ONE_CLASS = 1


def dice_score_one_class(prediction: torch.Tensor, target: torch.Tensor,
                         epsilon: float = 1e-6) -> float:
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
        float: DICE score averaged across batch
    """
    assert all(
        [value in ZERO_AND_ONE for value in torch.unique(prediction)])
    assert all([value in ZERO_AND_ONE for value in torch.unique(target)])
    assert prediction.shape[CHANNELS_DIM] == ONE_CLASS and \
           target.shape[CHANNELS_DIM] == ONE_CLASS
    all_but_batch_dims = list(range(1, target.dim()))
    intersection = (prediction * target).sum(dim=all_but_batch_dims)
    union = prediction.sum(dim=all_but_batch_dims) + \
                target.sum(dim=all_but_batch_dims)
    score = ((2. * intersection) / (union + epsilon))
    return torch.mean(score).item()
