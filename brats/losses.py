import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Compute per channel Dice Loss for single class.
    The loss will be average across channels.
    """

    def __init__(self, epsilon: float = 1e-5):
        """
        Args:
            epsilon: smooth factor to prevent division by zero
        """
        super(DiceLoss, self).__init__()
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
        intersection = (prediction * target).sum()

        loss = 1 - ((2. * intersection) / (prediction.sum() + target.sum() +
                                           self.epsilon))
        return loss

