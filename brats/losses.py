import torch
import torch.nn as nn

ZERO_AND_ONE = torch.tensor([0., 1.])
CHANNELS_DIM = 1
ONE_CLASS = 1


class DiceLossOneClass(nn.Module):
    """
    Compute Dice Loss for single class.
    The loss will be averaged across channels.
    """

    def __init__(self, epsilon: float = 1e-5):
        """
        Args:
            epsilon: smooth factor to prevent division by zero
        """
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: output of a network, expected to be already binarized.
                Dimensions - (Batch, Channels, Depth, Height, Width)
            target: labels.
                Dimensions - (Batch, Channels, Depth, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss
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
        loss = 1 - ((2. * intersection) / (union + self.epsilon))
        return torch.mean(loss)
