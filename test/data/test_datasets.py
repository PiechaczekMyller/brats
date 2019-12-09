import os

import pytest
import nibabel as nib
from brats.data import datasets
import numpy as np

INPUT_IMAGE_SHAPE = (10, 16, 16, 4)
IMAGES = 2


@pytest.fixture(scope="module")
def input_directory(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp("input_images")
    for i in range(IMAGES):
        image = np.zeros(INPUT_IMAGE_SHAPE, dtype=np.uint8)
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
