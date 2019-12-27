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

    def __init__(self, root: str, transform=None):
        self._files = list(os.scandir(root))
        self._transform = transform

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> typing.Any:
        scan = nib.load(self._files[idx].path)
        scan_array = scan.get_fdata()

        if self._transform:
            scan_array = self._transform(scan_array)

        return scan_array


class CombinedDataset(data.Dataset):
    """
    Takes two datasets of the same length and combines them,
    """

    def __init__(self, *datasets: data.Dataset):
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "Length of both datasets must match"
        self.datasets = datasets

    def __len__(self) -> int:
        return len(self.datasets[0])

    def __getitem__(self, idx: int) -> typing.Tuple[typing.Any, ...]:
        return tuple(dataset[idx] for dataset in self.datasets)
