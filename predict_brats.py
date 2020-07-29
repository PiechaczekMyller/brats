import argparse
import enum
import json
import os
import pathlib
import warnings
from time import time

import torch
import numpy as np
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.data.datasets import read_dataset_json
from brats.models import UNet3D
from brats.data import datasets
from brats.training.runners import run_inference

try:
    from apex import amp
except ImportError:
    warnings.warn("Apex ModuleNotFoundError, faked version used")
    from brats.training import fake_amp as amp


class Labels(enum.IntEnum):
    BACKGROUND = 0
    EDEMA = 1
    NON_ENHANCING = 2
    ENHANCING = 3


if __name__ == '__main__':
    brats_val_data = "/Users/szymek/Downloads/MICCAI_BraTS2020_ValidationData_stacked"
    out_dir = "/Users/szymek/Downloads/MICCAI_BraTS2020_ValidationData_stacked_preds"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda:0"

    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            trfs.Lambda(lambda x: torch.from_numpy(x)),
                                            trfs.Lambda(
                                                lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                            transformations.StandardizeVolumeWithFilter(0),
                                            trfs.Lambda(lambda x: x.float())
                                            ])

    volumes_paths = [x.path for x in os.scandir(brats_val_data)]
    volumes_set = datasets.NiftiFolder(volumes_paths, volumes_transformations)

    scans_loader = torch.utils.data.DataLoader(volumes_set, batch_size=1)

    model = UNet3D(4, 4).float()
    model.to(device)

    outputs = run_inference(model, scans_loader, device)
    for path, out in zip(volumes_paths, outputs):
        out_path = volumes_paths.replace(brats_val_data, out_dir)
        a=1
