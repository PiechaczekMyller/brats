import argparse
import os

import numpy as np
import torch
from ignite.contrib.handlers import TensorboardLogger, ProgressBar, global_step_from_engine
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms as trfs
from brats import transformations
from brats.data import datasets
from brats.losses import DiceLossOneClass
from brats.models import UNet3D


def score_function(engine):
    dice_loss = engine.state.metrics['dice_loss']
    return 1 - dice_loss


def get_volumes_transformations(input_size, device):
    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            trfs.Lambda(lambda x: x[3, :, :, :]),
                                            trfs.Lambda(lambda x: np.expand_dims(x, 0)),
                                            trfs.Lambda(lambda x: torch.from_numpy(x)),
                                            trfs.Lambda(
                                                lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x),
                                            transformations.ResizeVolumeTransformation(input_size),
                                            transformations.StandardizeVolume(),
                                            trfs.Lambda(lambda x: x.to(device))
                                            ])
    return volumes_transformations


def get_masks_transformations(input_size, device):
    masks_transformations = trfs.Compose([transformations.AddChannelDimToMaskTransformation(),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          transformations.BinarizationTransformation(),
                                          trfs.Lambda(
                                              lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x),
                                          transformations.ResizeVolumeTransformation(input_size),
                                          trfs.Lambda(lambda x: x.to(device))
                                          ])
    return masks_transformations


def get_metrics():
    return {'dice_loss': Loss(DiceLossOneClass())}


def attach_progress_bar(trainer):
    pbar = ProgressBar()
    pbar.attach(trainer)


def attach_tensorboard(log_dir, train_evaluator, validation_evaluator, trainer):
    tb_logger = TensorboardLogger(log_dir=os.path.join(log_dir, "tensorboard"), flush_secs=10)
    tb_logger.attach(train_evaluator,
                     log_handler=OutputHandler(tag="training",
                                               metric_names=[key for key in get_metrics()],
                                               global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(validation_evaluator,
                     log_handler=OutputHandler(tag="validation",
                                               metric_names=[key for key in get_metrics()],
                                               global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.EPOCH_COMPLETED)


def attach_periodic_checkpoint(engine, model, log_dir, n_saved):
    state_dict_checkpoint = ModelCheckpoint(os.path.join(log_dir, 'state_dicts'),
                                            filename_prefix='state_dict',
                                            save_interval=1,
                                            n_saved=n_saved,
                                            create_dir=True,
                                            save_as_state_dict=True)
    engine.add_event_handler(Events.EPOCH_COMPLETED, state_dict_checkpoint, {'unet3D': model})

    model_checkpoint = ModelCheckpoint(os.path.join(log_dir, 'models'),
                                       filename_prefix='model',
                                       save_interval=1,
                                       n_saved=n_saved,
                                       create_dir=True,
                                       save_as_state_dict=False)
    engine.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'unet3D': model})


def attach_best_checkpoint(engine, model, log_dir, score_function):
    best_state_dict_checkpoint = ModelCheckpoint(os.path.join(log_dir, 'best'),
                                                 filename_prefix='state_dict',
                                                 score_function=score_function,
                                                 n_saved=1,
                                                 create_dir=True,
                                                 save_as_state_dict=True)
    engine.add_event_handler(Events.EPOCH_COMPLETED, best_state_dict_checkpoint, {'unet3D': model})
    best_model_checkpoint = ModelCheckpoint(os.path.join(log_dir, 'best'),
                                            filename_prefix='model',
                                            score_function=score_function,
                                            n_saved=1,
                                            create_dir=True,
                                            save_as_state_dict=False)
    engine.add_event_handler(Events.EPOCH_COMPLETED, best_model_checkpoint, {'unet3D': model})


def attach_early_stopping(engine, patience, score_function):
    early_stoping = EarlyStopping(patience, score_function, engine)
    engine.add_event_handler(Events.COMPLETED, early_stoping)


def create_parser():
    parser = argparse.ArgumentParser(description='Train UNet 3D.')
    parser.add_argument('--volumes_path', type=str, required=True)
    parser.add_argument('--masks_path', type=str, required=True)
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

    return parser


def get_sets(volumes_path, masks_path, volumes_transformations, masks_transformations):
    volumes_set = datasets.NiftiFolder(volumes_path, volumes_transformations)
    masks_set = datasets.NiftiFolder(masks_path, masks_transformations)

    train_indeces = [1]
    # train_indeces = list(range(0, int(len(volumes_set) * args.train_valid_ratio)))
    # valid_indeces = list(range(int(len(volumes_set) * args.train_valid_ratio), len(volumes_set)))
    valid_indeces = [2]

    train_volumes_set = Subset(volumes_set, train_indeces)
    valid_volumes_set = Subset(volumes_set, valid_indeces)
    train_masks_set = Subset(masks_set, train_indeces)
    valid_masks_set = Subset(masks_set, valid_indeces)

    train_set = datasets.CombinedDataset(train_volumes_set, train_masks_set)
    valid_set = datasets.CombinedDataset(valid_volumes_set, valid_masks_set)
    return train_set, valid_set


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    volumes_transformations = get_volumes_transformations(args.input_size, args.device)
    masks_transformations = get_masks_transformations(args.input_size, args.device)

    train_set, valid_set = get_sets(args.volumes_path, args.masks_path, volumes_transformations, masks_transformations)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    model = UNet3D(args.in_channels, args.out_channels).double()
    model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    metrics = get_metrics()
    trainer = create_supervised_trainer(model, optimizer, DiceLossOneClass())
    train_evaluator = create_supervised_evaluator(model, metrics, args.device)
    validation_evaluator = create_supervised_evaluator(model, metrics, args.device)


    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(train_loader)
        validation_evaluator.run(valid_loader)


    @trainer.on(Events.EPOCH_COMPLETED)
    def print_train_results(engine):
        metrics = train_evaluator.state.metrics
        print(f"Training Results - Epoch: {trainer.state.epoch}  "
              f"Dice loss: {metrics['dice_loss']:.4f} "
              f"Dice: {1-metrics['dice_loss']:.4f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def print_validation_results(engine):
        metrics = validation_evaluator.state.metrics
        print(f"Validation Results - Epoch: {trainer.state.epoch}  "
              f"Dice loss: {metrics['dice_loss']:.4f} "
              f"Dice: {1-metrics['dice_loss']:.4f}")


    attach_progress_bar(trainer)
    attach_tensorboard(args.log_dir, train_evaluator, validation_evaluator, trainer)
    attach_periodic_checkpoint(validation_evaluator, model, args.log_dir, n_saved=args.epochs)
    attach_best_checkpoint(validation_evaluator, model, args.log_dir, score_function=score_function)
    attach_early_stopping(trainer, args.patience, score_function=score_function)

    trainer.run(train_loader, max_epochs=args.epochs)
