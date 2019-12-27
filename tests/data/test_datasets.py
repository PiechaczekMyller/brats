import os

import nibabel as nib
import numpy as np
import pytest
from torchvision.transforms import Lambda

from brats.data import datasets

INPUT_IMAGE_SHAPE = (10, 16, 16, 4)
IMAGES = 2


@pytest.fixture(scope="module")
def input_directory(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp("input_images")
    for i in range(IMAGES):
        image = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.uint8) + i
        nib_img = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(nib_img, os.path.join(temp_dir, f"image{i}.nii.gz"))
    return temp_dir


class TestNiftiFolder:
    def test_if_loads_images_from_directory(self, input_directory):
        dataset = datasets.NiftiFolder(input_directory)
        assert len(dataset) == IMAGES

    def test_if_loads_images_in_correct_shape(self, input_directory):
        dataset = datasets.NiftiFolder(input_directory)

        for idx in range(len(dataset)):
            assert dataset[idx].shape == INPUT_IMAGE_SHAPE

    def test_if_applies_transforms(self, input_directory):
        transform = Lambda(lambda x: np.pad(x, (2, 2)))
        dataset = datasets.NiftiFolder(input_directory, transform)

        for idx in range(len(dataset)):
            assert np.all(dataset[idx].shape == np.array(INPUT_IMAGE_SHAPE) + 4)


class TestCombinedDataset:
    def test_returns_correct_length(self, input_directory):
        dataset1 = datasets.NiftiFolder(input_directory)
        dataset2 = datasets.NiftiFolder(input_directory)
        dataset3 = datasets.NiftiFolder(input_directory)

        dataset = datasets.CombinedDataset(dataset1, dataset2, dataset3)
        assert len(dataset) == IMAGES

    def test_returns_correct_entries(self, input_directory):
        dataset1 = datasets.NiftiFolder(input_directory)
        dataset2 = datasets.NiftiFolder(input_directory)
        dataset3 = datasets.NiftiFolder(input_directory)

        dataset = datasets.CombinedDataset(dataset1, dataset2, dataset3)

        assert all(np.array_equal(entry, dataset1[0]) for entry in dataset[0])

    def test_if_raises_on_unequal_length(self, input_directory):
        dataset1 = datasets.NiftiFolder(input_directory)
        dataset2 = datasets.NiftiFolder(input_directory)
        dataset3 = datasets.NiftiFolder(input_directory)

        dataset2._files = [dataset2._files[0]]
        with pytest.raises(AssertionError):
            datasets.CombinedDataset(dataset1, dataset2, dataset3)
