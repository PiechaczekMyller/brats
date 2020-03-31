import argparse
import enum
import json
import os
import warnings
from time import time

import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms as trfs
from brats import transformations
from brats.data.datasets import read_dataset_json
from brats.losses import DiceLoss
from brats.metrics import DiceScore
from brats.models import UNet3D
from brats.data import datasets
from brats.training.runners import run_training_epoch, run_validation_epoch
from brats.training.stop_conditions import EarlyStopping
from brats.training.loggers import TensorboardLogger, BestModelLogger, BestStateDictLogger, ModelLogger, \
    StateDictsLogger, log_parameters

try:
    from apex import amp
except ImportError:
    warnings.warn("Apex ModuleNotFoundError, faked version used")
    from brats.training import fake_amp as amp


class Labels(enum.IntEnum):
    EDEMA = 0
    NON_ENHANCING = 1
    ENHANCING = 2


if __name__ == '__main__':

    def create_parser():
        parser = argparse.ArgumentParser(description='Train UNet 3D.')
        parser.add_argument('--dataset_json', type=str, required=True)
        parser.add_argument('--division_json', type=str, required=True)
        parser.add_argument('--log_dir', type=str, required=True)
        parser.add_argument('--device', type=str, required=True)
        parser.add_argument('--epochs', type=int, required=True)
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
                                            trfs.Lambda(lambda x: F.interpolate(x, size=args.input_size)),
                                            transformations.StandardizeVolume(),
                                            trfs.Lambda(lambda x: x.float()),
                                            ])
    masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          transformations.OneHotEncoding([0, 1, 2, 3]),
                                          trfs.Lambda(
                                              lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x),
                                          trfs.Lambda(lambda x: F.interpolate(x, size=args.input_size)),
                                          trfs.Lambda(lambda x: x.float())
                                          ])

    volumes_paths, masks_paths = read_dataset_json(args.dataset_json)
    volumes_set = datasets.NiftiFolder(volumes_paths, volumes_transformations)
    masks_set = datasets.NiftiFolder(masks_paths, masks_transformations)
    combined_set = datasets.CombinedDataset(volumes_set, masks_set)
    with open(args.division_json, "r") as division_file:
        indices = json.load(division_file)
    train_set = Subset(combined_set, indices["train"])
    valid_set = Subset(combined_set, indices["valid"])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)

    model = UNet3D(4, 3).float()
    model.to(args.device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    dice = DiceScore()
    metrics = {"dice": dice}

    last_model_logger = ModelLogger(os.path.join(args.log_dir, "models"))
    last_state_dict_logger = StateDictsLogger(os.path.join(args.log_dir, "models"))
    best_model_logger = BestModelLogger(os.path.join(args.log_dir, "best"))
    best_state_dict_logger = BestStateDictLogger(os.path.join(args.log_dir, "best"))
    tensorboard_logger = TensorboardLogger(args.log_dir)

    early_stopping = EarlyStopping(args.patience)

    log_parameters(args.log_dir, args)
    for epoch in range(args.epochs):
        time_0 = time()
        train_loss, train_metrics = run_training_epoch(model, train_loader, optimizer, criterion, metrics, args.device)
        valid_loss, valid_metrics = run_validation_epoch(model, valid_loader, criterion, metrics, args.device)
        print(f"Epoch: {epoch} "
              f"Train loss: {train_loss:.4f} "
              f"Valid loss: {valid_loss:.4f} "
              f"Valid dice edema: {valid_metrics['dice'][Labels.EDEMA]:.4f} "
              f"Valid dice non enhancing: {valid_metrics['dice'][Labels.NON_ENHANCING]:.4f} "
              f"Valid dice enhancing: {valid_metrics['dice'][Labels.ENHANCING]:.4f} "
              f"Time per epoch: {time()-time_0:.4f}s", flush=True)

        last_model_logger.log(model, "last_epoch")
        last_state_dict_logger.log(model, 0)
        best_model_logger.log(model, 3 - valid_loss)
        best_state_dict_logger.log(model, 3 - valid_loss)
        tensorboard_logger.log("loss/training_loss", train_loss, epoch)
        tensorboard_logger.log("loss/validation_loss", valid_loss, epoch)
        mean_dice = np.mean(
            [valid_metrics['dice'][Labels.EDEMA],
             valid_metrics['dice'][Labels.NON_ENHANCING],
             valid_metrics['dice'][Labels.ENHANCING]])
        tensorboard_logger.log("dice/dice_edema", valid_metrics['dice'][Labels.EDEMA], epoch)
        tensorboard_logger.log("dice/dice_non_enhancing", valid_metrics['dice'][Labels.NON_ENHANCING], epoch)
        tensorboard_logger.log("dice/dice_enhancing", valid_metrics['dice'][Labels.ENHANCING], epoch)
        tensorboard_logger.log("dice/mean_dice", mean_dice, epoch)

        if early_stopping.check_stop_condition(valid_loss):
            break
