import typing
from copy import deepcopy
from functools import singledispatch
import numpy as np
import torch
from torch.nn import functional as F

CHANNELS_IDX = 3


class HistogramMatchingTransformation:
    """
    Transformation performing histogram matching of volumes
    """

    def __init__(self, template: np.ndarray):
        """

        Args:
            template (np.ndarray): Template which will be used for histogram
                matching in future calls.
                Dimensions - (Height, Width, Depth, Channel)
        """
        self.template = template

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Args:
            volume (np.ndarray): Volume to be matched.
                Dimensions - (Height, Width, Depth, Channel)

        Returns:
             np.ndarray: Normalized volume with the same shape as input volume
        """
        assert self.template.ndim == volume.ndim
        result = np.zeros_like(volume)
        for channel in range(volume.shape[CHANNELS_IDX]):
            result[..., channel] = self.hist_match(volume[..., channel],
                                                   self.template[..., channel])
        return result

    @staticmethod
    def hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
            Adjust the pixel values of an image such that its histogram
            matches that of a target image

            Args:
                source (np.ndarray): Image to transform; the histogram is
                    computed over the flattened array
                template (np.ndarray): Template image; can have different
                    dimensions to source
            Returns:
                np.ndarray: The transformed output image
                    of the same shape as input
            """
        positive = source > 0
        result = np.zeros_like(source)
        source = source[positive].ravel()
        template = template[template > 0].ravel()
        # get the set of unique pixel values and their corresponding indices and
        # counts
        source_values, bin_idx, source_counts = np.unique(source,
                                                          return_inverse=True,
                                                          return_counts=True)
        template_values, template_counts = np.unique(template,
                                                     return_counts=True)
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        source_quantiles = np.cumsum(source_counts).astype(np.float64)
        source_quantiles /= source_quantiles[-1]
        template_quantiles = np.cumsum(template_counts).astype(np.float64)
        template_quantiles /= template_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(source_quantiles, template_quantiles,
                                    template_values)
        result[positive] = interp_t_values[bin_idx]
        return result


@singledispatch
def reorder(img: typing.Any):
    raise TypeError("Img should be either np.ndarray of torch.Tensor")


@reorder.register(np.ndarray)
def _(img: np.ndarray) -> np.ndarray:
    transformed = np.moveaxis(img, [0, 1, 2, 3], [2, 3, 1, 0])
    return transformed


@reorder.register(torch.Tensor)
def _(img: torch.Tensor) -> torch.Tensor:
    transformed = img.permute(3, 2, 0, 1)
    return transformed


class NiftiToTorchDimensionsReorderTransformation:
    """
    Changes dimensions order from nifti (H,W,D,C) to torch convention (C,D,H,W).
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        assert len(img.shape) == 4, "Tensor should have 4 dimensions (H,W,D,C)"

        transformed = reorder(img)
        return transformed


@singledispatch
def add_channel_dim(img: typing.Any):
    raise TypeError("Img should be either np.ndarray of torch.Tensor")


@add_channel_dim.register(np.ndarray)
def _(img: np.ndarray) -> np.ndarray:
    transformed = np.expand_dims(img, 3)
    return transformed


@add_channel_dim.register(torch.Tensor)
def _(img: torch.Tensor) -> torch.Tensor:
    transformed = torch.unsqueeze(img, 0)
    return transformed


class AddChannelDimToMaskTransformation:
    """
    Adds additional channel dimension
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        assert img.ndim == 3, "Img should have 3 dimensions (D,H,W)"
        transformed = add_channel_dim(img)
        return transformed


@singledispatch
def binarize(img: typing.Any):
    return TypeError("Img should be either np.ndarray of torch.Tensor")


@binarize.register(np.ndarray)
def _(img: np.ndarray) -> np.ndarray:
    transformed = img.copy()
    transformed[transformed > 0] = 1
    return transformed


@binarize.register(torch.Tensor)
def _(img: torch.Tensor) -> torch.Tensor:
    transformed = img.clone().detach()
    transformed[transformed > 0] = 1
    return transformed


class BinarizationTransformation:
    """
    Changes image to binary mask, all values above 1 would be changed to 1.
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        transformed = binarize(img)
        return transformed


class ResizeVolumeTransformation:
    """
    Resizes the volume spatial dimensions (H,W) to the given size.
    For details, see nn.functional.interpolate.
    """

    def __init__(self, size: typing.Union[int, tuple]):
        self.size = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndim == 4, "Tensor should have 4 dimensions (C,D,H,W)"

        out = F.interpolate(tensor, size=self.size)  # The resize operation on tensor.
        return out


class StandardizeVolume:
    """
    Performs standardization on the volume
    (mean -> 0, std -> 1 with standard metrics)
    Standardization is done on the volumetric dimensions,
    if multiple channels are given, standardization is done channel-wise.
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        means = tensor.mean(dim=[1, 2, 3], keepdims=True)
        stds = tensor.std(dim=[1, 2, 3], keepdims=True)
        transformed = (tensor - means) / stds
        return transformed
