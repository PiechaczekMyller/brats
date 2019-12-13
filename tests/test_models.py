import pytest
from brats import models

import os
import torch

import pytest
import nibabel as nib
from brats.data import datasets
import numpy as np


class TestUnet3D:
    @pytest.mark.parametrize("in_channels, out_channels, batch_size", [(3, 1, 1),
                                                                       (3, 3, 1),
                                                                       (1, 3, 1),
                                                                       (1, 1, 1),
                                                                       (1, 1, 2),
                                                                       (1, 1, 5)])
    def test_if_returns_correct_shape(self, in_channels, out_channels, batch_size):
        model = models.UNet3D(in_channels, out_channels, 2)
        tensor = torch.zeros((batch_size, in_channels, 32, 64, 64))
        assert model(tensor).shape == (batch_size, out_channels, 32, 64, 64)


class TestUnet3DBlock:
    @pytest.mark.parametrize("in_channels, out_channels, batch_size", [(3, 1, 1),
                                                                       (3, 3, 1),
                                                                       (1, 3, 1),
                                                                       (1, 1, 1),
                                                                       (1, 1, 2),
                                                                       (1, 1, 5)])
    def test_if_returns_correct_shape(self, in_channels, out_channels, batch_size):
        model = models.UNet3DBlock(in_channels, out_channels)
        tensor = torch.zeros((batch_size, in_channels, 5, 10, 10))
        assert model(tensor).shape == (batch_size, out_channels, 5, 10, 10)
