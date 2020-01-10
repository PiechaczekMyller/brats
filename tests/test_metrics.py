import torch
import pytest
import numpy as np

import brats.metrics as metrics

BATCH_DIMS = (2, 1, 3, 3, 3)
CHANNEL_DIM = 1


class TestDiceScoreOneClass:
    metric = metrics.DiceScoreOneClass()

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 0, atol=1.e-4)

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
        assert np.isclose(self.metric(images.unsqueeze(dim=CHANNEL_DIM),
                                      target.unsqueeze(dim=CHANNEL_DIM)),
                           result, atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert self.metric(images, target).shape == torch.Size([])


class TestRecallScore:
    metric = metrics.RecallScore()

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.334),
                              (torch.tensor([[1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.6667),
                              (torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.334),
                              (torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.6667)
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(self.metric(images.unsqueeze(dim=CHANNEL_DIM),
                                      target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert self.metric(images, target).shape == torch.Size([])


class TestPrecisionScore:
    metric = metrics.PrecisionScore()

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric(images, target), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.5),
                              (torch.tensor([[1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.6667),
                              (torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]), 0.5),
                              (torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.6667)
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(self.metric(images.unsqueeze(dim=CHANNEL_DIM),
                                      target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert self.metric(images, target).shape == torch.Size([])


class TestFScore:
    metric_beta1 = metrics.FScore(beta=1)
    metric_beta2 = metrics.FScore(beta=2)

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric_beta1(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(self.metric_beta1(images, target), 0, atol=1.e-4)

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
    def test_if_f1_returns_expected_values(self, images, target, result):
        assert np.isclose(self.metric_beta1(images.unsqueeze(dim=CHANNEL_DIM),
                                            target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.3577),
                              (torch.tensor([[1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 0.6667),
                              (torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.3577),
                              (torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1], [1, 1, 0, 1]]),
                               0.6667)
                              ])
    def test_if_f2_returns_expected_values(self, images, target, result):
        assert np.isclose(self.metric_beta2(images.unsqueeze(dim=CHANNEL_DIM),
                                            target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert self.metric_beta1(images, target).shape == torch.Size([])


class TestHausdorffDistance95:
    metric = metrics.HausdorffDistance95()

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0, 1, 1, 0]]),
                               torch.tensor([[1, 1, 0, 1]]), 1),
                              (torch.tensor([[1, 0, 0, 0]]),
                               torch.tensor([[0, 0, 0, 1]]), 1.414),
                              (torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]),
                               torch.tensor([[0, 0, 1, 0], [0, 0, 0, 0]]),
                               2),
                              (torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]]),
                               torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]]),
                               3.162)
                              ])
    def test_if_returns_correct_values(self, images, target, result):
        images = images.view(1, 1, 1, 2, -1)
        target = target.view(1, 1, 1, 2, -1)
        assert np.isclose(self.metric(images, target), result, atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert self.metric(images, target).shape == torch.Size([])
