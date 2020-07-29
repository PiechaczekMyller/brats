import argparse
import enum
import json
import os
import pathlib
import warnings

import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.losses import DiceLoss
from brats.metrics import DiceScore
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

    def create_parser():
        parser = argparse.ArgumentParser(description='Train UNet 3D.')
        parser.add_argument('--dataset_path', type=str, required=True)
        parser.add_argument('--division_json', type=str, required=True)
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--model_path', type=str, required=True)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument('--patience', type=int, default=10)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--input_size', type=int, default=240)
        parser.add_argument('--use_amp', dest='use_amp', action='store_true')
        parser.set_defaults(use_amp=False)
        return parser


    parser = create_parser()
    args = parser.parse_args()

    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            trfs.Lambda(lambda x: torch.from_numpy(x)),
                                            trfs.Lambda(
                                                lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                            transformations.StandardizeVolumeWithFilter(0),
                                            trfs.Lambda(lambda x: x.float())
                                            ])
    masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          transformations.OneHotEncoding([0, 1, 2, 3]),
                                          trfs.Lambda(
                                              lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x),
                                          trfs.Lambda(lambda x: x.float())
                                          ])
    common_transformations = transformations.ComposeCommon(
        [transformations.RandomCrop((args.input_size, args.input_size))])

    with open(args.division_json) as division_json:
        division = json.load(division_json)

    train_patients = division["train"]
    valid_patients = division["valid"]
    test_patients = division["test"]

    valid_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in valid_patients]
    valid_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in valid_patients]

    test_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in test_patients]
    test_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in test_patients]

    valid_volumes_set = datasets.NiftiFolder(valid_volumes_paths, volumes_transformations)
    valid_mask_set = datasets.NiftiFolder(valid_masks_paths, masks_transformations)
    valid_set = datasets.CombinedDataset(valid_volumes_set, valid_mask_set, transform=common_transformations)

    test_volumes_set = datasets.NiftiFolder(test_volumes_paths, volumes_transformations)
    test_mask_set = datasets.NiftiFolder(test_masks_paths, masks_transformations)
    test_set = datasets.CombinedDataset(test_volumes_set, test_mask_set, transform=common_transformations)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = torch.load(args.model_path)
    model.to(args.device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    dice = DiceScore()
    metrics = {"dice": dice}

    valid_outputs = run_inference(model, valid_loader, args.device)
    test_outputs = run_inference(model, valid_loader, args.device)

    valid_outputs_path = os.path.join(args.log_dir, 'outputs', 'valid')
    test_outputs_path = os.path.join(args.log_dir, 'outputs', 'test')

    pathlib.Path(valid_outputs_path).mkdir(parents=True, exist_ok=True)
    for output, input_path in zip(valid_outputs, valid_volumes_set._files):
        torch.save(output, os.path.join(valid_outputs_path, os.path.basename(input_path)))

    pathlib.Path(test_outputs_path).mkdir(parents=True, exist_ok=True)
    for output, input_path in zip(test_outputs, test_volumes_set._files):
        torch.save(output, os.path.join(test_outputs_path, os.path.basename(input_path)))
