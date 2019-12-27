from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import torch
import numpy as np
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.metrics import Loss, Average
from torch import optim
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.data import datasets
from brats.losses import DiceLossOneClass
from brats.models import UNet3D

train_images_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/train/images"
train_masks_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/train/masks"
valid_images_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/valid/images"
valid_masks_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/valid/masks"
device = "cpu"
images_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                       trfs.Lambda(lambda x: x[3, :, :, :]),
                                       trfs.Lambda(lambda x: np.expand_dims(x, 0)),
                                       trfs.Lambda(lambda x: torch.from_numpy(x)),
                                       trfs.Lambda(
                                           lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                       transformations.ResizeVolumeTransformation(16),
                                       transformations.StandardizeVolume()
                                       ])
masks_transformations = trfs.Compose([transformations.AddChannelDimToMaskTransformation(),
                                      transformations.NiftiToTorchDimensionsReorderTransformation(),
                                      trfs.Lambda(lambda x: torch.from_numpy(x)),
                                      transformations.BinarizationTransformation(),
                                      trfs.Lambda(lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x),
                                      trfs.Lambda(lambda x: F.interpolate(x, size=16))
                                      ])
train_images_set = datasets.NiftiFolder(train_images_path, images_transformations)
train_masks_set = datasets.NiftiFolder(train_masks_path, masks_transformations)
valid_images_set = datasets.NiftiFolder(valid_images_path, images_transformations)
valid_mask_set = datasets.NiftiFolder(valid_masks_path, masks_transformations)

train_set = datasets.CombinedDataset(train_images_set, train_masks_set)
valid_set = datasets.CombinedDataset(valid_images_set, valid_mask_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True)

model = UNet3D(1, 1).double()

optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = create_supervised_trainer(model, optimizer, DiceLossOneClass())
metrics = {'dice_loss': Loss(DiceLossOneClass())}

train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
validation_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

pbar = ProgressBar()
pbar.attach(trainer)


@trainer.on(Events.EPOCH_COMPLETED)
def compute_metrics(engine):
    train_evaluator.run(train_loader)
    validation_evaluator.run(valid_loader)


tb_logger = TensorboardLogger(log_dir="./logs", flush_secs=10)

tb_logger.attach(train_evaluator,
                 log_handler=OutputHandler(tag="training",
                                           metric_names=["dice_loss", "dice"],
                                           another_engine=trainer),
                 event_name=Events.EPOCH_COMPLETED)

tb_logger.attach(validation_evaluator,
                 log_handler=OutputHandler(tag="validation",
                                           metric_names=["dice_loss", "dice"],
                                           another_engine=trainer),
                 event_name=Events.EPOCH_COMPLETED)

model_checkpoint = ModelCheckpoint('./logs', 'model_', save_interval=2, n_saved=2, create_dir=True)
trainer.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {'mymodel': model})


def score_function(engine):
    dice_loss = engine.state.metrics['dice_loss']
    return 1 - dice_loss


early_stoping = EarlyStopping(10, score_function, trainer)
validation_evaluator.add_event_handler(Events.COMPLETED, early_stoping)

trainer.run(train_loader, max_epochs=50)
