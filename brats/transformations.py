import numpy as np

CHANNELS = 3


class HistogramMatchingTransform:
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
        for channel in range(volume.shape[CHANNELS]):
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
        nonzero = source > 0
        result = np.zeros_like(source)
        source = source[source > 0].ravel()
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
        result[nonzero] = interp_t_values[bin_idx]
        return result
