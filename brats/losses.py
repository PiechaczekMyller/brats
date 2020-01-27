import torch
import torch.nn as nn

import brats.functional as F

ONE_CLASS = 1


class DiceLoss(nn.Module):
    """
    Compute Dice Loss calculated as 1 - dice_score for each class and sum the
    result..
    The loss will be averaged across batch elements.
    """

    def __init__(self, epsilon: float = 1e-6):
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
            target: labels. The classes (channels) should be one-hot encoded
                Dimensions - (Batch, Channels, Depth, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss
        """
        return torch.sum(1 - F.dice(prediction, target, self.epsilon))
