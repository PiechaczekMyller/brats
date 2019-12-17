import os

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

    def __len__(self):
        return len(self._files)

    def __getitem__(self, item):
        scan = nib.load(self._files[item].path)
        scan_array = scan.get_fdata()

        if self._transform:
            scan_array = self._transform(scan_array)

        return scan_array


class CombinedDataset(data.Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, item):
        return self.dataset1[item], self.dataset2[item]
