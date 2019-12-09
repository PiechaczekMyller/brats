import os

from torch.utils import data
import nibabel as nib

from torchvision.datasets import ImageFolder


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
