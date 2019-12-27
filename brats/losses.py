import torch
import torch.nn as nn

from brats.metrics import dice_score_one_class

ONE_CLASS = 1


class DiceLossOneClass(nn.Module):
    """
    Compute Dice Loss for single class, calculated as 1 - dice_score.
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
            target: labels.
                Dimensions - (Batch, Channels, Depth, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss
        """
        loss = 1 - dice_score_one_class(prediction, target)
        return torch.mean(loss)
