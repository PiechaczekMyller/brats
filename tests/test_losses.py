import torch
import pytest
import numpy as np

from brats.losses import DiceLossOneClass

BATCH_DIMS = (2, 5, 5, 5)


class TestDiceLossOneClass:
    def test_if_returns_0_for_perfect_fit(self):
        images = torch.ones(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(DiceLossOneClass()(images, target), 0, atol=1.e-4)

    def test_if_returns_1_for_worst_fit(self):
        images = torch.zeros(*BATCH_DIMS)
        target = torch.ones(*BATCH_DIMS)
        assert np.isclose(DiceLossOneClass()(images, target), 1, atol=1.e-4)
