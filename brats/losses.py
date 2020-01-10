import torch
import torch.nn as nn

from brats.metrics import DiceScoreOneCLass

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
        self.dice_score = DiceScoreOneCLass()

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
        return 1 - self.dice_score(prediction, target)
