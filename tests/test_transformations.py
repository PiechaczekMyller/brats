import pytest
import numpy as np

from brats.transformations import HistogramMatchingTransform

INPUT_VOLUME_SHAPE_4D = (10, 10, 10, 3)
INPUT_VOLUME_SHAPE_3D = (10, 7, 5)
TEMPLATE_VOLUME_SHAPE_4D = (17, 7, 5, 2)


class TestHistogramMatchingTransform:
    def test_if_returns_with_same_shape(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        transformation = HistogramMatchingTransform(image)
        assert image.shape == transformation(image).shape

    def test_if_returns_same_for_identical_images(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        transformation = HistogramMatchingTransform(image)
        assert np.all(image == transformation(image))

    def test_if_4_dimensional_array_should_be_passed(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_3D)
        transformation = HistogramMatchingTransform(image)
        with pytest.raises(IndexError):
            transformation(image)

    def test_if_template_has_to_have_same_ndim_as_input_image(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        template = np.random.rand(*INPUT_VOLUME_SHAPE_3D)
        transformation = HistogramMatchingTransform(template)
        with pytest.raises(AssertionError):
            transformation(image)

    def test_if_template_and_input_have_to_have_same_no_of_channels(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        template = np.random.rand(*TEMPLATE_VOLUME_SHAPE_4D)
        transformation = HistogramMatchingTransform(template)
        with pytest.raises(IndexError):
            transformation(image)

    @pytest.mark.parametrize("template, image, result",
                             [(np.arange(25).reshape((5, 5, 1, 1)),
                               np.ones((5, 5, 1, 1)),
                               np.ones((5, 5, 1, 1)) * 24),
                              (np.array([[0.5, 0.9], [0.7, 0.9]]).reshape(
                                  (2, 2, 1, 1)),
                               np.array([[0.1, 0.2], [0.4, 0.1]]).reshape(
                                   (2, 2, 1, 1)),
                               np.array([[0.7, 0.8], [0.9, 0.7]]).reshape(
                                   (2, 2, 1, 1)))])
    def test_if_returns_correct_values(self, template, image, result):
        transformation = HistogramMatchingTransform(template)
        assert np.all(transformation(image) == result)
