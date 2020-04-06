import abc
import typing

import torch

import brats.functional as F

ONE_CLASS = 1
CLASS_DIM = 1


class Loss(abc.ABC):
    """
    Interface for loss
    """

    @abc.abstractmethod
    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiceLoss(Loss):
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


class NLLLossOneHot(Loss):
    """
    Compute negative log-likelihood loss for one hot encoded targets.
    """

    def __init__(self, epsilon: float = 1e-12, *args, **kwargs):
        """

        Args:
            epsilon: Epsilon value to prevent log(0)
            *args: Arguments accepted by the torch.nn.NLLLoss() constructor
            **kwargs: Keyword arguments accepted by the torch.nn.NLLLoss()
                    constructor
        """
        super().__init__()
        self.epsilon = epsilon
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
        prediction = torch.log(prediction + self.epsilon)
        return self.nll_loss(prediction, target)


class ComposedLoss(Loss):
    """
    Compute weighted sum on provided losses
    """

    def __init__(self, losses: typing.List[Loss],
                 weights: typing.List[float] = None):
        """
        Args:
            losses: A list of losses to use.
            weights: Weights for each loss.
        """
        if weights is not None:
            assert len(losses) == len(
                weights), "Number of losses should be the same as number of weights"
            self.weights = weights
        else:
            self.weights = [1. for _ in range(len(losses))]
        self.losses = losses

    def __call__(self, prediction: torch.Tensor,
                 target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: output of a network, expected to be already binarized.
                Dimensions - (Batch, Channels, Depth, Height, Width)
            target: labels. The classes (channels) should be one-hot encoded
                Dimensions - (Batch, Channels, Depth, Height, Width)
        Returns:
            Weighted sum of provided losses
        """
        weighted_sum = torch.tensor(0., device=prediction.device)
        for i, loss in enumerate(self.losses):
            weighted_sum += loss(prediction, target) * self.weights[i]
        return weighted_sum
