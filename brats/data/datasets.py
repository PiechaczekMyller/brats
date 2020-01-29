import json
import os
import typing

import nibabel as nib
from torch.utils import data


class NiftiFolder(data.Dataset):
    """
    A custom loader for .nii.gz files in a single folder.
    Each file should contain a scan of a single patient in one or more modalities.
    E.g.:
    scans/patient000.nii.gz
    scans/patient001.nii.gz
    scans/patient002.nii.gz
    where file scans/patient000.nii.gz contains scan of the patient 001 in 4 modalities:
    T1, T1gd, T2w, Flair
    (Note that the order of the modalities doesn't matter, however it should be consistent for whole dataset)
    """

    def __init__(self, paths: typing.List[str], transform: typing.List[typing.Callable] = None):
        self._files = paths
        self._transform = transform

    @classmethod
    def from_dir(cls, root: str, transforms: typing.Callable = None):
        files = [entry.path for entry in os.scandir(root)]
        return NiftiFolder(files, transforms)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> typing.Any:
        scan = nib.load(self._files[idx])
        scan_array = scan.get_fdata()

        if self._transform:
            scan_array = self._transform(scan_array)

        return scan_array


class CombinedDataset(data.Dataset):
    """
    Takes multiple datasets of the same length and combines them.
    On `__getitem__(n)` it returns a tuple containing nth element of each dataset.
    """

    def __init__(self, *datasets: data.Dataset):
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "Length of all datasets must be the same"
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> typing.Tuple[typing.Any, ...]:
        return tuple(dataset[idx] for dataset in self.datasets)


def read_dataset_json(path_to_json):
    """
    Reads pairs of images and masks from json file.
    :param path_to_json: Path to the file from decathlon challange
    :return: Tuple with list of paths to images and list of path to masks
    """
    with open(path_to_json, "r") as json_file:
        json_dict = json.load(json_file)
    root = os.path.dirname(path_to_json)
    images_paths = [os.path.join(root, line["image"].replace("./", "")) for line in json_dict["training"]]
    masks_paths = [os.path.join(root, line["label"].replace("./", "")) for line in json_dict["training"]]
    return images_paths, masks_paths
