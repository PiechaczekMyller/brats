import torch
import pytest
import numpy as np

import brats.losses as losses

BATCH_DIMS = (2, 1, 5, 5, 5)
CHANNEL_DIM = 1
BATCH_DIM = 0


class TestDiceLossOneClass:
    def test_if_returns_0_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(losses.DiceLoss()(images, target), 0, atol=1.e-4)

    def test_if_returns_1_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(losses.DiceLoss()(images, target), 1, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0., 1., 1., 0]]),
                               torch.tensor([[1., 1., 0., 1]]), 0.6),
                              (torch.tensor([[1., 1., 1., 0]]),
                               torch.tensor([[1., 1., 0., 1]]), 0.334),
                              (torch.tensor([[0., 1., 1., 0], [0., 1., 1., 0]]),
                               torch.tensor([[1., 1., 0., 1], [1., 1., 0., 1]]), 0.6),
                              (torch.tensor([[1., 1., 1., 0], [1., 1., 1., 0]]),
                               torch.tensor([[1., 1., 0., 1], [1., 1., 0., 1]]),
                               0.334)
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(losses.DiceLoss()(images.unsqueeze(dim=CHANNEL_DIM),
                                            target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-2)


class TestNLLLossOneHot:
    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0., 0.9]]),
                               torch.tensor([[0., 1.]]), 0.3412),
                              (torch.tensor([[0., 0.9], [0., 0.9]]),
                               torch.tensor([[0., 1.], [0, 1]]), 0.3412),
                              (torch.tensor([[0., 0.9, 0], [0.9, 0., 0.], [0., 0., 0.9]]),
                               torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]]), 0.8951)])
    def test_if_returns_correct_values(self, images, target, result):
        images = torch.nn.Softmax(dim=1)(images)
        assert np.isclose(losses.NLLLossOneHot()(images, target), result,
                          atol=1.e-4)


class TestDiceWithNLL:
    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0., 0.9]]),
                               torch.tensor([[0., 1.]]), 1.5101),
                              (torch.tensor([[0., 0.9], [0., 0.9]]),
                               torch.tensor([[0., 1.], [0., 1.]]), 1.5101),
                              (torch.tensor([[0., 0.9, 0], [0.0, 0., 1.], [0., 0., 0.9]]),
                               torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]]), 2.8629),
                              (torch.tensor([[0., 0.9, 0, 0.1], [0.0, 0.8, 0., 1.]]),
                               torch.tensor([[0., 1., 0., 0.], [0., 0., 0., 1.]]), 4.2893)
                             ])
    def test_if_returns_correct_values(self, images, target, result):
        images = images.unsqueeze(dim=2)
        target = target.unsqueeze(dim=2)
        images = torch.nn.Softmax(dim=1)(images)
        assert np.isclose(losses.DiceWithNLLOneHot()(images, target), result,
                          atol=1.e-4)
