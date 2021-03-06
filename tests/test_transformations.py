import torch

import pytest
import numpy as np
from brats import transformations as trfs

INPUT_VOLUME_SHAPE_4D = (10, 10, 10, 3)
INPUT_VOLUME_SHAPE_3D = (10, 7, 5)
TEMPLATE_VOLUME_SHAPE_4D = (17, 7, 5, 2)


class TestHistogramMatchingTransformation:
    def test_if_returns_with_same_shape(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        transformation = trfs.HistogramMatchingTransformation(image)
        assert image.shape == transformation(image).shape

    def test_if_returns_same_for_identical_images(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        transformation = trfs.HistogramMatchingTransformation(image)
        assert np.all(image == transformation(image))

    def test_if_4_dimensional_array_should_be_passed(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_3D)
        transformation = trfs.HistogramMatchingTransformation(image)
        with pytest.raises(IndexError):
            transformation(image)

    def test_if_template_has_to_have_same_ndim_as_input_image(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        template = np.random.rand(*INPUT_VOLUME_SHAPE_3D)
        transformation = trfs.HistogramMatchingTransformation(template)
        with pytest.raises(AssertionError):
            transformation(image)

    def test_if_template_and_input_have_to_have_same_no_of_channels(self):
        image = np.random.rand(*INPUT_VOLUME_SHAPE_4D)
        template = np.random.rand(*TEMPLATE_VOLUME_SHAPE_4D)
        transformation = trfs.HistogramMatchingTransformation(template)
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
        transformation = trfs.HistogramMatchingTransformation(template)
        assert np.all(transformation(image) == result)


class TestNiftiToTorchDimensionsReorderTransformation:
    def test_if_returns_correct_tensors(self):
        transformation = trfs.NiftiToTorchDimensionsReorderTransformation()
        input = torch.zeros([4, 5, 10, 3])
        expected = torch.zeros([3, 10, 4, 5])
        assert transformation(input).shape == expected.shape

    def test_if_returns_correct_ndarray(self):
        transformation = trfs.NiftiToTorchDimensionsReorderTransformation()
        input = np.zeros([4, 5, 10, 3])
        expected = np.zeros([3, 10, 4, 5])
        assert transformation(input).shape == expected.shape

    @pytest.mark.parametrize("input",
                             [torch.zeros((1, 2, 3, 4, 5)),
                              torch.zeros((1, 2, 3)),
                              torch.zeros((1, 2)),
                              torch.zeros(1),
                              np.zeros((1, 2, 3, 4, 5)),
                              np.zeros((1, 2, 3)),
                              np.zeros((1, 2)),
                              np.zeros(1)
                              ])
    def test_if_raises_on_incorrect_shape(self, input):
        transformation = trfs.NiftiToTorchDimensionsReorderTransformation()
        with pytest.raises(AssertionError):
            transformation(input)


class TestAddChannelDimToMaskTransformation:
    def test_if_returns_correct_tensors(self):
        transformation = trfs.AddChannelDimToMaskTransformation()
        input = torch.zeros([4, 5, 6])
        expected = torch.zeros([1, 4, 5, 6])
        assert transformation(input).shape == expected.shape

    def test_if_returns_correct_ndarray(self):
        transformation = trfs.AddChannelDimToMaskTransformation()
        input = np.zeros([4, 5, 6])
        expected = np.zeros([4, 5, 6, 1])
        assert transformation(input).shape == expected.shape

    @pytest.mark.parametrize("input",
                             [torch.zeros((1, 2, 3, 4, 5)),
                              torch.zeros((1, 2, 3, 4)),
                              torch.zeros((1, 2)),
                              torch.zeros(1),
                              np.zeros((1, 2, 3, 4, 5)),
                              np.zeros((1, 2, 3, 4)),
                              np.zeros((1, 2)),
                              np.zeros(1)
                              ])
    def test_if_raises_on_incorrect_shape(self, input):
        transformation = trfs.AddChannelDimToMaskTransformation()
        with pytest.raises(AssertionError):
            transformation(input)


class TestBinarizationTransformation:
    def test_if_binarize_tensor(self):
        transformation = trfs.BinarizationTransformation()
        input = torch.zeros([4, 5, 6, 7])
        input[1, :, :, :] = 1
        input[2, :, :, :] = 2
        input[3, :, :, :] = 3
        out = transformation(input)
        assert torch.all(torch.eq(out.unique(),
                                  torch.tensor([0., 1.])))

    def test_if_binarize_ndarray(self):
        transformation = trfs.BinarizationTransformation()
        input = np.zeros([4, 5, 6, 7])
        input[1, :, :, :] = 1
        input[2, :, :, :] = 2
        input[3, :, :, :] = 3
        out = transformation(input)
        assert np.all(np.array_equal(np.unique(out),
                                     np.array([0., 1.])))


class TestResizeVolumeTransformation:
    @pytest.mark.parametrize("input, size, expected",
                             [(torch.zeros((1, 1, 4, 4)), 4, torch.zeros((1, 1, 4, 4))),
                              (torch.zeros((1, 1, 4, 4)), 2, torch.zeros((1, 1, 2, 2))),
                              (torch.zeros((1, 1, 4, 4)), 3, torch.zeros((1, 1, 3, 3))),
                              (torch.zeros((1, 1, 4, 4)), 8, torch.zeros((1, 1, 8, 8))),
                              (torch.zeros((1, 1, 4, 4)), 9, torch.zeros((1, 1, 9, 9))),
                              (torch.zeros((1, 1, 4, 4)), (4, 4), torch.zeros((1, 1, 4, 4))),
                              (torch.zeros((1, 1, 4, 4)), (2, 2), torch.zeros((1, 1, 2, 2))),
                              (torch.zeros((1, 1, 4, 4)), (3, 3), torch.zeros((1, 1, 3, 3))),
                              (torch.zeros((1, 1, 4, 4)), (8, 8), torch.zeros((1, 1, 8, 8))),
                              (torch.zeros((1, 1, 4, 4)), (9, 9), torch.zeros((1, 1, 9, 9))),
                              (torch.zeros((1, 1, 4, 4)), (1, 3), torch.zeros((1, 1, 1, 3))),
                              (torch.zeros((1, 1, 4, 4)), (3, 1), torch.zeros((1, 1, 3, 1))),
                              (torch.zeros((1, 1, 4, 4)), (12, 11), torch.zeros((1, 1, 12, 11))),
                              (torch.zeros((1, 1, 4, 4)), (11, 12), torch.zeros((1, 1, 11, 12))),
                              ]
                             )
    def test_if_returns_correct_tensors(self, input, size, expected):
        transformation = trfs.ResizeVolumeTransformation(size)
        assert transformation(input).shape == expected.shape

    @pytest.mark.parametrize("input",
                             [torch.zeros((1, 2, 3, 4, 5)),
                              torch.zeros((1, 2, 3)),
                              torch.zeros((1, 2)),
                              torch.zeros(1)])
    def test_if_raises_on_incorrect_shape(self, input):
        transformation = trfs.ResizeVolumeTransformation(2)
        with pytest.raises(AssertionError):
            transformation(input)


class TestStandardizeVolume:
    def test_if_standardize_tensor(self):
        transformation = trfs.StandardizeVolume()
        input = torch.randn((3, 15, 10, 10)) * 2.3 + 1.4
        out = transformation(input)
        assert float(out.mean()) == pytest.approx(0, abs=0.001)
        assert float(out.std()) == pytest.approx(1, abs=0.001)

    def test_if_epsilon_handles_nans(self):
        input = torch.zeros(3, 5, 4, 4)

        transformation = trfs.StandardizeVolume(epsilon=0)
        assert torch.any(torch.isnan(transformation(input)))
        transformation.epsilon = 1e-6
        assert not torch.any(torch.isnan(transformation(input)))


class TestStandardizeVolumeWithFilter:
    def test_if_standardize_tensor(self):
        transformation = trfs.StandardizeVolumeWithFilter(0)
        input = torch.randn((3, 15, 10, 10)) * 2.3 + 1.4
        out = transformation(input)
        assert float(out.mean()) == pytest.approx(0, abs=0.001)
        assert float(out.std()) == pytest.approx(1, abs=0.001)

    def test_if_epsilon_handles_nans(self):
        input = torch.zeros(3, 5, 4, 4)

        transformation = trfs.StandardizeVolumeWithFilter(1, epsilon=0)
        assert torch.any(torch.isnan(transformation(input)))
        transformation.epsilon = 1e-6
        assert not torch.any(torch.isnan(transformation(input)))

    @pytest.mark.parametrize("input, value_to_filter, unique",
                             [(torch.tensor([1, 2, 3, 4]), 2, torch.tensor([1, 3, 4])),
                              (torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), 2, torch.tensor([1, 3, 4])),
                              (torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]), 5, torch.tensor([1, 2, 3, 4]))
                              ])
    def test_if_filter_removes_values(self, input, value_to_filter, unique):
        transformation = trfs.StandardizeVolumeWithFilter(value_to_filter)
        assert torch.all(torch.unique(transformation._filter(input)) == unique)


class TestOneHotEncoding:
    @pytest.mark.parametrize("input, n_classes",
                             [(torch.cat([torch.ones((1, 10, 4, 4)) * x for x in range(n)]), n) for n in range(2, 30)])
    def test_if_returns_correct_tensors(self, input, n_classes):
        transformation = trfs.OneHotEncoding([n for n in range(n_classes)])
        assert transformation(input).shape == (n_classes, 10, 4, 4)


class TestRandomCrop:
    @pytest.mark.parametrize("img, size, desired",
                             [(torch.zeros((4, 3, 50, 50)), (10, 10), torch.zeros((4, 3, 10, 10))),
                              (torch.zeros((4, 3, 50, 50)), (12, 12), torch.zeros((4, 3, 12, 12))),
                              (torch.zeros((4, 3, 240, 240)), (128, 128), torch.zeros((4, 3, 128, 128))),
                              (torch.zeros((1, 3, 240, 240)), (128, 128), torch.zeros((1, 3, 128, 128))),
                              (torch.zeros((1, 1, 240, 240)), (128, 128), torch.zeros((1, 1, 128, 128))),
                              (np.zeros((4, 3, 50, 50)), (10, 10), np.zeros((4, 3, 10, 10))),
                              (np.zeros((4, 3, 50, 50)), (12, 12), np.zeros((4, 3, 12, 12))),
                              (np.zeros((4, 3, 240, 240)), (128, 128), np.zeros((4, 3, 128, 128))),
                              (np.zeros((1, 3, 240, 240)), (128, 128), np.zeros((1, 3, 128, 128))),
                              (np.zeros((1, 1, 240, 240)), (128, 128), np.zeros((1, 1, 128, 128)))
                              ])
    def test_if_returns_correct_shapes(self, img, size, desired):
        transformation = trfs.RandomCrop(size)
        transformed = transformation(img)[0]
        assert transformed.shape == desired.shape

    @pytest.mark.parametrize("img1, img2, img3",
                             [(torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 9))),
                              (torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 9, 10))),
                              (torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 9)), torch.zeros((4, 3, 10, 10))),
                              (torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 9, 10)), torch.zeros((4, 3, 10, 10))),
                              (torch.zeros((4, 3, 10, 9)), torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))),
                              (torch.zeros((4, 3, 9, 10)), torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))),
                              (np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 9))),
                              (np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 9, 10))),
                              (np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 9)), np.zeros((4, 3, 10, 10))),
                              (np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 9, 10)), np.zeros((4, 3, 10, 10))),
                              (np.zeros((4, 3, 10, 9)), np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10))),
                              (np.zeros((4, 3, 9, 10)), np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10)))
                              ])
    def test_if_raises_on_mismatched_shapes(self, img1, img2, img3):
        transformation = trfs.RandomCrop((10, 10))
        with pytest.raises(AssertionError):
            transformation(img1, img2, img3)

    @pytest.mark.parametrize("imgs",
                             [
                                 [torch.zeros((4, 3, 10, 10))],
                                 [torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))],
                                 [torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10)),
                                  torch.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10))]
                             ])
    def test_if_processes_variadic_arguments(self, imgs):
        transformation = trfs.RandomCrop((5, 5))
        desired = torch.zeros((4, 3, 5, 5))
        assert all([img.shape == desired.shape for img in transformation(*imgs)])

    @pytest.mark.parametrize("imgs",
                             [
                                 [torch.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10))],
                                 [torch.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))],
                                 [np.zeros((4, 3, 10, 10)), np.zeros((4, 3, 10, 10)), torch.zeros((4, 3, 10, 10))]
                             ])
    def test_if_processes_variadic_types(self, imgs):
        transformation = trfs.RandomCrop((5, 5))
        desired_shape = (4, 3, 5, 5)
        assert all([tuple(img.shape) == desired_shape for img in transformation(*imgs)])
