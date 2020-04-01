import torch
import torch.nn as nn

import brats.functional as F

ONE_CLASS = 1
CLASS_DIM = 1


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


class NLLLossOneHot(nn.Module):
    """
    Compute negative log-likelihood loss for one hot encoded targets.
    """

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Arguments accepted by the torch.nn.NLLLoss() constructor
            **kwargs: Keyword arguments accepted by the torch.nn.NLLLoss()
                    constructor
        """
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(*args, **kwargs)

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: output of a network, expected to be already binarized.
                Dimensions - (Batch, Channels, Depth, Height, Width)
            target: labels. The classes (channels) should be one-hot encoded
                Dimensions - (Batch, Channels, Depth, Height, Width)
        Returns:
            torch.Tensor: Computed NLL Loss
        """
        target = torch.argmax(target, dim=CLASS_DIM)
        prediction = torch.log(prediction)
        return self.nll_loss(prediction, target)


class DiceWithNLL(nn.Module):
    """
    Compute Dice loss and negative log-likelihood and return the unweighted sum.
    """

    def __init__(self, epsilon: float = 1e-6, *args, **kwargs):
        """
        Args:
            epsilon: smooth factor to prevent division by zero
            *args: Arguments accepted by the torch.nn.NLLLoss() constructor
            **kwargs: Keyword arguments accepted by the torch.nn.NLLLoss()
                    constructor
        """
        super().__init__()
        self.epsilon = epsilon
        self.nll_loss = NLLLossOneHot(*args, **kwargs)

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: output of a network, expected to be already binarized.
                Dimensions - (Batch, Channels, Depth, Height, Width)
            target: labels. The classes (channels) should be one-hot encoded
                Dimensions - (Batch, Channels, Depth, Height, Width)

        Returns:
            torch.Tensor: Computed Dice Loss summed with Cross Entropy Loss
        """
        return torch.sum(
            1 - F.dice(prediction, target, self.epsilon)) + self.nll_loss(
            prediction, target)
