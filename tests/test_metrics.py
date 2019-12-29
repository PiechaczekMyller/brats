import torch
import pytest
import numpy as np

import brats.metrics as metrics

BATCH_DIMS = (2, 1, 3, 3, 3)
CHANNEL_DIM = 1


class TestDiceScoreOneClass:
    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.dice_score_one_class(images, target), 1,
                              atol=1.e-4))

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.dice_score_one_class(images, target), 0,
                              atol=1.e-4))

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
        assert all(np.isclose(
            metrics.dice_score_one_class(images.unsqueeze(dim=CHANNEL_DIM),
                                         target.unsqueeze(dim=CHANNEL_DIM)),
            result,
            atol=1.e-3))


class TestRecallScore:
    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.recall_score(images, target), 1,
                              atol=1.e-4))

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.recall_score(images, target), 0,
                              atol=1.e-4))

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
        assert all(np.isclose(
            metrics.recall_score(images.unsqueeze(dim=CHANNEL_DIM),
                                 target.unsqueeze(dim=CHANNEL_DIM)),
            result,
            atol=1.e-3))


class TestPrecisionScore:
    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.precision_score(images, target), 1,
                              atol=1.e-4))

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.precision_score(images, target), 0,
                              atol=1.e-4))

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
        assert all(np.isclose(
            metrics.precision_score(images.unsqueeze(dim=CHANNEL_DIM),
                                    target.unsqueeze(dim=CHANNEL_DIM)),
            result,
            atol=1.e-3))


class TestFScore:
    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.f_score(images, target, beta=1), 1,
                              atol=1.e-4))

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert all(np.isclose(metrics.f_score(images, target, beta=1), 0,
                              atol=1.e-4))

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
        assert all(np.isclose(
            metrics.f_score(images.unsqueeze(dim=CHANNEL_DIM),
                            target.unsqueeze(dim=CHANNEL_DIM), beta=1),
            result,
            atol=1.e-3))

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
        assert all(np.isclose(
            metrics.f_score(images.unsqueeze(dim=CHANNEL_DIM),
                            target.unsqueeze(dim=CHANNEL_DIM), beta=2),
            result,
            atol=1.e-3))


class TestHausdorffDistance95:
    @pytest.mark.parametrize("batch_size, result", [((2, 1, 3, 3, 3), (2, 3)),
                                                    ((5, 1, 2, 2, 2), (5, 2)),
                                                    ((1, 1, 1, 1, 1), (1, 1))])
    def test_if_returns_correct_shapes(self, batch_size, result):
        images = torch.ones(*batch_size)
        target = torch.ones(*batch_size)
        assert metrics.hausdorff_distance95(images, target).shape == result

    @pytest.mark.parametrize("batch_size, result", [((2, 1, 3, 3, 3), (2, 3)),
                                                    ((5, 1, 2, 2, 2), (5, 2)),
                                                    ((1, 1, 1, 1, 1), (1, 1))])
    def test_if_mean_max_returns_correct_shapes(self, batch_size, result):
        images = torch.ones(*batch_size)
        target = torch.ones(*batch_size)
        assert len(metrics.hausdorff_distance95_mean_max(images, target)[0]) == \
               result[0]
