"""
Initializing weights.

"""
import torch
import numpy as np

from torch import nn


def get_upsample_filter(size):
    """
    Make a 2D bilinear kernel suitable for upsampling.

    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


def initialize_lapsrn(model: nn.Module, negative_slope: float = 0.2):
    """
    Initialization as described in "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks".
    This is in-place operation.

    :param model: Model whose weights should be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data, a=negative_slope)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.ConvTranspose2d):
            c1, c2, h, w = m.weight.data.size()
            weight = get_upsample_filter(h)
            m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
            if m.bias is not None:
                m.bias.data.zero_()


def initialize_default(model: nn.Module):
    """
    Simple xavier initialization. This is in-place operation.

    :param model: Model whose weights should be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
