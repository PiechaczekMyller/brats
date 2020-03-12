import torch
import pytest
import numpy as np

import brats.functional as F

BATCH_DIMS = (2, 1, 3, 3, 3)
CHANNEL_DIM = 1


class TestDice:

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.dice(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.dice(images, target), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_values_for_one_class(self, images, target, result):
        assert np.isclose(F.dice(images.unsqueeze(dim=CHANNEL_DIM),
                                 target.unsqueeze(dim=CHANNEL_DIM)),
                          result, atol=1.e-3)

    @pytest.mark.parametrize("input_shape, result", [((1, 2, 1, 1, 1), 2),
                                                     ((2, 3, 1, 1, 1), 3),
                                                     ((3, 3, 2, 2, 2), 3)])
    def test_if_returns_correct_number_of_classes(self, input_shape, result):
        images = torch.ones(*input_shape)
        target = torch.ones(*input_shape)
        assert len(F.dice(images, target)) == result

    @pytest.mark.parametrize("images, target, classes, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 3, torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 4,
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_value_for_multiclass(self, images, target, classes, result):
        images = images.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        target = target.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        result = [result for _ in range(classes)]
        assert np.all(np.isclose(F.dice(images, target), result, atol=1.e-3))


class TestRecallScore:

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.recall(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.recall(images, target), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.334])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.334])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(F.recall(images.unsqueeze(dim=CHANNEL_DIM),
                                   target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    @pytest.mark.parametrize("input_shape, result", [((1, 2, 1, 1, 1), 2),
                                                     ((2, 3, 1, 1, 1), 3),
                                                     ((3, 3, 2, 2, 2), 3)])
    def test_if_returns_correct_number_of_classes(self, input_shape, result):
        images = torch.ones(*input_shape)
        target = torch.ones(*input_shape)
        assert len(F.recall(images, target)) == result

    @pytest.mark.parametrize("images, target, classes, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.334])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 3, torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.334])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 4,
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_value_for_multiclass(self, images, target, classes, result):
        images = images.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        target = target.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        result = [result for _ in range(classes)]
        assert np.all(np.isclose(F.recall(images, target), result, atol=1.e-3))


class TestPrecisionScore:

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.precision(images, target), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.precision(images, target), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.5])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.5])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_values(self, images, target, result):
        assert np.isclose(F.precision(images.unsqueeze(dim=CHANNEL_DIM),
                                      target.unsqueeze(dim=CHANNEL_DIM)),
                          result,
                          atol=1.e-3)

    @pytest.mark.parametrize("input_shape, result", [((1, 2, 1, 1, 1), 2),
                                                     ((2, 3, 1, 1, 1), 3),
                                                     ((3, 3, 2, 2, 2), 3)])
    def test_if_returns_correct_number_of_classes(self, input_shape, result):
        images = torch.ones(*input_shape)
        target = torch.ones(*input_shape)
        assert len(F.precision(images, target)) == result

    @pytest.mark.parametrize("images, target, classes, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.5])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 3, torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.5])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 4,
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_value_for_multiclass(self, images, target, classes, result):
        images = images.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        target = target.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        result = [result for _ in range(classes)]
        assert np.all(np.isclose(F.precision(images, target), result, atol=1.e-3))


class TestFScore:

    def test_if_returns_1_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.f_score(images, target, beta=1), 1, atol=1.e-4)

    def test_if_returns_0_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(F.f_score(images, target, beta=1), 0, atol=1.e-4)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.6667]))
                              ])
    def test_if_f1_returns_expected_values(self, images, target, result):
        assert np.isclose(F.f_score(images.unsqueeze(dim=CHANNEL_DIM),
                                    target.unsqueeze(dim=CHANNEL_DIM), beta=1),
                          result,
                          atol=1.e-3)

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.3577])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.3577])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]),
                               torch.Tensor([0.6667]))
                              ])
    def test_if_f2_returns_expected_values(self, images, target, result):
        assert np.isclose(F.f_score(images.unsqueeze(dim=CHANNEL_DIM),
                                    target.unsqueeze(dim=CHANNEL_DIM), beta=2),
                          result,
                          atol=1.e-3)

    @pytest.mark.parametrize("input_shape, result", [((1, 2, 1, 1, 1), 2),
                                                     ((2, 3, 1, 1, 1), 3),
                                                     ((3, 3, 2, 2, 2), 3)])
    def test_if_returns_correct_number_of_classes(self, input_shape, result):
        images = torch.ones(*input_shape)
        target = torch.ones(*input_shape)
        assert len(F.f_score(images, target, 1)) == result

    @pytest.mark.parametrize("images, target, classes, result",
                             [(torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), 3, torch.Tensor([0.6667])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 2, torch.Tensor([0.4])),
                              (torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]), 4,
                               torch.Tensor([0.6667]))
                              ])
    def test_if_returns_expected_value_for_multiclass(self, images, target, classes, result):
        images = images.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        target = target.unsqueeze(dim=CHANNEL_DIM).repeat(1, classes, 1)
        result = [result for _ in range(classes)]
        assert np.all(np.isclose(F.f_score(images, target, 1), result, atol=1.e-3))


class TestHausdorffDistance95:

    @pytest.mark.parametrize("images, target, result",
                             [(torch.tensor([[0.0, 0.0, 0.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([0])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[0.0, 0.0, 0.0, 0.0]]), torch.Tensor([0])),
                              (torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
                               torch.tensor([[1.0, 1.0, 0.0, 1.0]]), torch.Tensor([1])),
                              (torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                               torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.Tensor([1.414])),
                              (torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                               torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                               torch.Tensor([2])),
                              (torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                               torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                               torch.Tensor([3.162]))
                              ])
    def test_if_returns_correct_values(self, images, target, result):
        images = images.view(1, 1, 1, 2, -1)
        target = target.view(1, 1, 1, 2, -1)
        assert np.isclose(F.hausdorff95(images, target), result, atol=1.e-3)

    def test_if_returns_tensor_with_shape_0(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert F.hausdorff95(images, target).shape == torch.Size([])
