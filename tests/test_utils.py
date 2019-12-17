import torch
import pytest

from brats.utils import calculate_union, calculate_intersection

VOLUME_DIMS = (3, 3, 3)


class TestIntersection:
    @pytest.mark.parametrize("batch_size", [(2,), (5,), (10,)])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        assert len(calculate_intersection(x1, x2, dim=VOLUME_DIMS)) == \
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
        assert calculate_intersection(x1, x2) == result


class TestUnion:
    @pytest.mark.parametrize("batch_size", [(2, ), (5, ), (10, )])
    def test_if_returns_value_for_each_element_in_batch(self, batch_size):
        x1 = torch.ones(batch_size + VOLUME_DIMS)
        x2 = torch.ones(batch_size + VOLUME_DIMS)
        assert len(calculate_union(x1, x2, dim=VOLUME_DIMS)) == batch_size[0]

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
        assert calculate_union(x1, x2) == result
