import argparse
import enum
import json
import os
import warnings
from time import time
import dill
import mlflow
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms as trfs

from torch.cuda.amp import GradScaler
from brats import transformations
from brats.data.datasets import read_dataset_json
from brats.losses import DiceLoss, ComposedLoss, NLLLossOneHot
from brats.metrics import DiceScore
from brats.models import UNet3D
from brats.data import datasets
from brats.training.runners import run_training_epoch, run_validation_epoch
from brats.training.stop_conditions import EarlyStopping
from brats.training.loggers import TensorboardLogger, BestModelLogger, BestStateDictLogger, ModelLogger, \
    StateDictsLogger, log_parameters, log_git_info


class Labels(enum.IntEnum):
    BACKGROUND = 0
    EDEMA = 1
    NON_ENHANCING = 2
    ENHANCING = 3


class FromNumpy:
    def __call__(self, x):
        return torch.from_numpy(x)


class ExpandDims:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return np.expand_dims(x, self.dim)


class PadTo160:
    def __call__(self, x):
        return F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x


class ToFloat:
    def __call__(self, x):
        return x.float()


if __name__ == '__main__':

    def create_parser():
        parser = argparse.ArgumentParser(description='Train UNet 3D.')
        parser.add_argument('--dataset_path', type=str, required=True)
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

    mlflow.set_tracking_uri('http://kp-desktops2.kplabs.pl:5069')
    mlflow.set_experiment("UNet3D")
    mlflow.start_run(run_name="Pancreas and cancer segmentation")

    mlflow.log_param('log dir', args.log_dir)
    mlflow.log_param('epochs', args.epochs)
    mlflow.log_param('batch_size', args.batch_size)
    mlflow.log_param('patience', args.patience)
    mlflow.log_param('learning_rate', args.learning_rate)
    mlflow.log_param('input_size', args.input_size)
    mlflow.log_param('dataset_path', args.dataset_path)
    mlflow.log_param('division_json', args.division_json)


    class FromNumpy:
        def __call__(self, x):
            return torch.from_numpy(x)


    class ExpandDims:
        def __init__(self, dim):
            self.dim = dim

        def __call__(self, x):
            return np.expand_dims(x, self.dim)


    class PadTo160:
        def __call__(self, x):
            a= F.pad(x, [0, 0, 0, 0, 16-(x.shape[1] % 16), 0]) if x.shape[1] % 16 != 0 else x
            return a


    class ToFloat:
        def __call__(self, x):
            return x.float()


    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            FromNumpy(),
                                            PadTo160(),
                                            transformations.StandardizeVolumeWithFilter(0),
                                            ToFloat(),
                                            transformations.ResizeVolumeTransformation(args.input_size)
                                            ])
    masks_transformations = trfs.Compose([ExpandDims(3),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          FromNumpy(),
                                          transformations.OneHotEncoding([0, 1, 2]),
                                          PadTo160(),
                                          ToFloat(),
                                          transformations.ResizeVolumeTransformation(args.input_size)
                                          ])
    # common_transformations = transformations.ComposeCommon(
    #     [])

    with open(args.division_json) as division_json:
        division = json.load(division_json)

    train_patients = division["train"]
    valid_patients = division["valid"]
    test_patients = division["test"]

    train_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in train_patients]
    train_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in train_patients]

    valid_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in valid_patients]
    valid_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in valid_patients]

    test_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in test_patients]
    test_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in test_patients]

    train_volumes_set = datasets.NiftiFolder(train_volumes_paths, volumes_transformations)
    train_mask_set = datasets.NiftiFolder(train_masks_paths, masks_transformations)
    # train_set = datasets.CombinedDataset(train_volumes_set, train_mask_set, transform=common_transformations)
    train_set = datasets.CombinedDataset(train_volumes_set, train_mask_set)

    valid_volumes_set = datasets.NiftiFolder(valid_volumes_paths, volumes_transformations)
    valid_mask_set = datasets.NiftiFolder(valid_masks_paths, masks_transformations)
    # valid_set = datasets.CombinedDataset(valid_volumes_set, valid_mask_set, transform=common_transformations)
    valid_set = datasets.CombinedDataset(valid_volumes_set, valid_mask_set)

    test_volumes_set = datasets.NiftiFolder(test_volumes_paths, volumes_transformations)
    test_mask_set = datasets.NiftiFolder(test_masks_paths, masks_transformations)
    # test_set = datasets.CombinedDataset(test_volumes_set, test_mask_set, transform=common_transformations)
    test_set = datasets.CombinedDataset(test_volumes_set, test_mask_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = UNet3D(1, 3).float()
    model.to(args.device)

    criterion = DiceLoss(epsilon=1e-4)
    optimizer = optim.Adam(model.parameters(), args.learning_rate, eps=1e-4)
    scaler = GradScaler()

    dice = DiceScore(epsilon=1e-4)
    metrics = {"dice": dice}

    last_model_logger = ModelLogger(os.path.join(args.log_dir, "models"))
    last_state_dict_logger = StateDictsLogger(os.path.join(args.log_dir, "models"))
    best_model_logger = BestModelLogger(os.path.join(args.log_dir, "best"))
    best_state_dict_logger = BestStateDictLogger(os.path.join(args.log_dir, "best"))
    tensorboard_logger = TensorboardLogger(args.log_dir)

    early_stopping = EarlyStopping(args.patience)

    log_parameters(args.log_dir, args)
    log_git_info(args.log_dir)
    for epoch in range(args.epochs):
        time_0 = time()
        train_loss, train_metrics = run_training_epoch(model, train_loader, optimizer, scaler, criterion, metrics,
                                                       args.device)
        valid_loss, valid_metrics = run_validation_epoch(model, valid_loader, criterion, metrics, args.device)
        print(f"Epoch: {epoch} "
              f"Train loss: {train_loss:.4f} "
              f"Valid loss: {valid_loss:.4f} "
              f"Valid dice background: {valid_metrics['dice'][Labels.BACKGROUND]:.4f} "
              f"Valid dice pancreas: {valid_metrics['dice'][Labels.EDEMA]:.4f} "
              f"Valid dice cancer: {valid_metrics['dice'][Labels.NON_ENHANCING]:.4f} "
              # f"Valid dice enhancing: {valid_metrics['dice'][Labels.ENHANCING]:.4f} "
              f"Time per epoch: {time() - time_0:.4f}s", flush=True)

        mlflow.log_metric("train_loss", train_loss)

        last_model_logger.log(model, "last_epoch")
        last_state_dict_logger.log(model, 0)
        best_model_logger.log(model, 3 - valid_loss)
        best_state_dict_logger.log(model, 3 - valid_loss)

        mean_dice = np.mean(
            [valid_metrics['dice'][Labels.BACKGROUND],
             valid_metrics['dice'][Labels.EDEMA],
             valid_metrics['dice'][Labels.NON_ENHANCING],
             # valid_metrics['dice'][Labels.ENHANCING]
             ])

        mean_dice_no_background = np.mean([
            valid_metrics['dice'][Labels.EDEMA],
            valid_metrics['dice'][Labels.NON_ENHANCING],
            # valid_metrics['dice'][Labels.ENHANCING]
        ])

        mlflow.log_metric("train_loss", train_loss, epoch)
        mlflow.log_metric("valid_loss", valid_loss, epoch)
        mlflow.log_metric("mean_dice", mean_dice, epoch)
        mlflow.log_metric("dice_background", valid_metrics['dice'][Labels.BACKGROUND], epoch)
        mlflow.log_metric("dice_pancreas", valid_metrics['dice'][Labels.EDEMA], epoch)
        mlflow.log_metric("dice_cancer", valid_metrics['dice'][Labels.NON_ENHANCING], epoch)
        # mlflow.log_metric("dice_enhancing", valid_metrics['dice'][Labels.ENHANCING], epoch)
        mlflow.log_metric("time_per_epoch", time() - time_0, epoch)

        if early_stopping.check_stop_condition(valid_loss):
            break
    test_loss, test_metrics = run_validation_epoch(model, test_loader, criterion, metrics, args.device)
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_dice_background", test_metrics['dice'][Labels.BACKGROUND])
    mlflow.log_metric("test_dice_pancreas", test_metrics['dice'][Labels.EDEMA])
    mlflow.log_metric("test_dice_cancer", test_metrics['dice'][Labels.NON_ENHANCING])
    # mlflow.log_metric("test_dice_enhancing", test_metrics['dice'][Labels.ENHANCING])
