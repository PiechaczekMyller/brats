import torch
import torch.nn as nn

CHANNEL, HEIGHT, WIDTH = 1, 2, 3


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
            prediction: output of a network, expected to be already normalized.
                Dimensions - (Batch, Channels, Height, Width)
            target: labels.
                Dimensions - (Batch, Channels, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss

        """
        prediction = prediction.contiguous()
        target = target.contiguous()
        intersection = (prediction * target).sum(dim=(CHANNEL, HEIGHT, WIDTH))

        loss = 1 - ((2. * intersection) / (
                prediction.sum(dim=(CHANNEL, HEIGHT, WIDTH)) +
                target.sum(dim=(CHANNEL, HEIGHT, WIDTH)) + self.epsilon))
        return torch.mean(loss)
