import torch
import pytest
import numpy as np

from brats.metrics import dice_score_one_class

BATCH_DIMS = (2, 1, 5, 5, 5)
CHANNEL_DIM = 1


class TestDiceScoreOneClass:
    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(dice_score_one_class(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(dice_score_one_class(images, target), 0, atol=1.e-4)

    def test_if_works_for_non_binary_data(self):
        images = torch.ones(*BATCH_DIMS) * 0.5
        target = torch.ones(*BATCH_DIMS)
        with pytest.raises(AssertionError):
            dice_score_one_class(images, target)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.4),
                              (torch.tensor([[1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.6667),
                              (torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]), 0.4),
                              (torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.6667)
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(
            dice_score_one_class(images.unsqueeze(dim=CHANNEL_DIM),
                                 target.unsqueeze(dim=CHANNEL_DIM)),
            result,
            atol=1.e-3)
