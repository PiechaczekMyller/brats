import torch
import pytest
import numpy as np

import brats.utils as utils

VOLUME_DIMS = (3, 3, 3)


class TestIsBinary:
    def test_if_returns_false_for_all_non_binary_values(self):
        input = torch.ones(VOLUME_DIMS) * 0.5
        assert not utils.is_binary(input)

    def test_if_returns_false_for_one_non_binary_value(self):
        input = torch.ones(VOLUME_DIMS)
        input[0, 0, 0] = 0.2
        assert not utils.is_binary(input)

    def test_if_returns_true_for_only_1s(self):
        assert utils.is_binary(torch.ones(VOLUME_DIMS))

    def test_if_returns_true_for_only_0s(self):
        assert utils.is_binary(torch.zeros(VOLUME_DIMS))


class TestHasOnlyZeros:

    def test_if_returns_true_for_only_zeros(self):
        input = np.zeros(VOLUME_DIMS)
        assert utils.has_only_zeros(input)

    def test_if_returns_false_for_different_than_zero(self):
        input = np.zeros(VOLUME_DIMS)
        input[0, 0, 0] = 1
        assert not utils.has_only_zeros(input)

    def test_if_returns_false_for_only_ones(self):
        input = np.ones(VOLUME_DIMS)
        assert not utils.has_only_zeros(input)


class TestCalculateFalseNegatives:
    @pytest.mark.parametrize("batch_size", [(2,), (5,), (10,)])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        all_but_batch_dims = list(range(1, x2.dim()))
        assert len(utils.calculate_false_negatives(x1, x2,
                                                   dim=all_but_batch_dims)) \
               == batch_size[0]

    @pytest.mark.parametrize("x1, x2, result", [(torch.tensor([1, 1]),
                                                 torch.tensor([1, 1.]), 0),
                                                (torch.tensor([1, 0, 1, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 1),
                                                (torch.tensor([1, 1, 0, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 0),
                                                (torch.tensor([[0, 0], [0, 0]]),
                                                 torch.tensor([[1, 1], [1, 1]]),
                                                 4)])
    def test_if_returns_correct_values(self, x1, x2, result):
        assert utils.calculate_false_negatives(x1, x2) == result


class TestCalculateFalsePositives:
    @pytest.mark.parametrize("batch_size", [(2,), (5,), (10,)])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        all_but_batch_dims = list(range(1, x2.dim()))
        assert len(utils.calculate_false_positives(x1, x2, dim=all_but_batch_dims)) == \
               batch_size[0]

    @pytest.mark.parametrize("x1, x2, result", [(torch.tensor([1, 1]),
                                                 torch.tensor([1, 1.]), 0),
                                                (torch.tensor([1, 0, 1, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 2),
                                                (torch.tensor([1, 1, 0, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 1),
                                                (torch.tensor([[1, 1], [1, 1]]),
                                                 torch.tensor([[0, 0], [0, 0]]),
                                                 4)])
    def test_if_returns_correct_values(self, x1, x2, result):
        assert utils.calculate_false_positives(x1, x2) == result


class TestIntersection:
    @pytest.mark.parametrize("batch_size", [(2,), (5,), (10,)])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        all_but_batch_dims = list(range(1, x2.dim()))
        assert len(utils.calculate_intersection(x1, x2, dim=all_but_batch_dims)) == \
               batch_size[0]

    @pytest.mark.parametrize("x1, x2, result", [(torch.tensor([1, 1]),
                                                 torch.tensor([1, 1.]), 2),
                                                (torch.tensor([1, 0, 1, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 0),
                                                (torch.tensor([1, 1, 0, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 1),
                                                (torch.tensor([[1, 0], [0, 1]]),
                                                 torch.tensor([[1, 0], [0, 1]]),
                                                 2)])
    def test_if_returns_correct_values(self, x1, x2, result):
        assert utils.calculate_intersection(x1, x2) == result


class TestUnion:
    @pytest.mark.parametrize("batch_size", [(2,), (5,), (10,)])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        all_but_batch_dims = list(range(1, x2.dim()))
        assert len(utils.calculate_union(x1, x2, dim=all_but_batch_dims)) == \
               batch_size[0]

    @pytest.mark.parametrize("x1, x2, result", [(torch.tensor([1, 1]),
                                                 torch.tensor([1, 1.]), 4),
                                                (torch.tensor([1, 0, 1, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 3),
                                                (torch.tensor([1, 1, 0, 0]),
                                                 torch.tensor([0, 1, 0, 0]), 3),
                                                (torch.tensor([[1, 0], [0, 1]]),
                                                 torch.tensor([[1, 0], [0, 1]]),
                                                 4)])
    def test_if_returns_correct_values(self, x1, x2, result):
        assert utils.calculate_union(x1, x2) == result
