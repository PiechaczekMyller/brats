import argparse
import json
import os

import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms as trfs
from brats import transformations
from brats.data.datasets import read_dataset_json
from brats.losses import DiceLoss
from brats.models import UNet3D
from brats.data import datasets
from brats.training.runners import run_training_epoch, run_validation_epoch
from brats.training.stop_conditions import EarlyStopping
from brats.training.loggers import TensorboardLogger, BestModelLogger, BestStateDictLogger, ModelLogger, \
    StateDictsLogger, log_parameters

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
        parser.add_argument('--in_channels', type=int, default=1)
        parser.add_argument('--out_channels', type=int, default=1)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--train_valid_ratio', type=float, default=0.8)
        parser.add_argument('--input_size', type=int, default=240)
        parser.add_argument('--progress_bar', type=int, default=0)

        return parser


    parser = create_parser()
    args = parser.parse_args()

    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            trfs.Lambda(lambda x: x[3, :, :, :]),
                                            trfs.Lambda(lambda x: np.expand_dims(x, 0)),
                                            trfs.Lambda(lambda x: torch.from_numpy(x)),
                                            trfs.Lambda(
                                                lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                            trfs.Lambda(lambda x: F.interpolate(x, size=32)),
                                            transformations.StandardizeVolume(),
                                            trfs.Lambda(lambda x: x.float()),
                                            ])
    masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          transformations.BinarizationTransformation(),
                                          trfs.Lambda(
                                              lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                          trfs.Lambda(lambda x: F.interpolate(x, size=32)),
                                          trfs.Lambda(lambda x: x.float()),
                                          ])

    volumes_paths, masks_paths = read_dataset_json(args.dataset_json)
    volumes_set = datasets.NiftiFolder(volumes_paths, volumes_transformations)
    masks_set = datasets.NiftiFolder(masks_paths, masks_transformations)
    combined_set = datasets.CombinedDataset(volumes_set, masks_set)
    with open(args.division_json, "r") as division_file:
        indeces = json.load(division_file)
    train_set = Subset(combined_set, indeces["train"])
    valid_set = Subset(combined_set, indeces["valid"])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    model = UNet3D(1, 1).float()
    model.to(args.device)
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate)

    last_model_logger = ModelLogger(os.path.join(args.log_dir, "models"))
    last_state_dict_logger = StateDictsLogger(os.path.join(args.log_dir, "models"))
    best_model_logger = BestModelLogger(os.path.join(args.log_dir, "best"))
    best_state_dict_logger = BestStateDictLogger(os.path.join(args.log_dir, "best"))
    tensorboard_logger = TensorboardLogger(args.log_dir)

    early_stopping = EarlyStopping(args.patience)

    log_parameters(args.log_dir, args)
    for epoch in range(args.epochs):
        train_loss = run_training_epoch(model, train_loader, optimizer, criterion, args.device)
        valid_loss = run_validation_epoch(model, valid_loader, criterion, args.device)
        print(f"Epoch: {epoch}  "
              f"Train loss: {train_loss:.4f} "
              f"Valid loss: {valid_loss:.4f}", flush=True)

        last_model_logger.log(model, 0)
        last_state_dict_logger.log(model, 0)
        best_model_logger.log(model, 1 - valid_loss)
        best_state_dict_logger.log(model, 1 - valid_loss)
        tensorboard_logger.log("Training loss", train_loss, epoch)
        tensorboard_logger.log("Validation loss", valid_loss, epoch)

        early_stopping.update(valid_loss)
        if early_stopping.check_stop_condition():
            break
