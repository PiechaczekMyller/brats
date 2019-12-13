import torch
import torch.nn as nn


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
        super(DiceLossOneClass, self).__init__()
        self.epsilon = epsilon

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: output of a network, expected to be already binarized.
                Dimensions - (Batch, Channels, Height, Width)
            target: labels.
                Dimensions - (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss
        """
        all_but_batch_dims = list(range(1, target.dim()))
        prediction = prediction.contiguous()
        target = target.contiguous()
        intersection = (prediction * target).sum(dim=all_but_batch_dims)
        loss = 1 - ((2. * intersection) / (
                prediction.sum(dim=all_but_batch_dims) +
                target.sum(dim=all_but_batch_dims) + self.epsilon))
        return torch.mean(loss)
