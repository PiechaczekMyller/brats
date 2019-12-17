import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.losses import DiceLossOneClass
from brats.models import UNet3D
from brats.data import datasets
from brats.training.observers import TensorboardLogger, PyTorchWeightsSaver
from brats.training.runners import TrainingEpochRunner, ValidationEpochRunner
from brats.training.trainers import PyTorchTrainer

train_images_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/train/images"
train_masks_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/train/masks"
valid_images_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/valid/images"
valid_masks_path = fr"/Users/szymek/Documents/Task01_BrainTumour/small/valid/masks"

images_transformations = trfs.Compose([transformations.NiftiOrderTransformation(),
                                       trfs.Lambda(lambda x: x[3, :, :, :]),
                                       trfs.Lambda(lambda x: np.expand_dims(x, 0)),
                                       trfs.Lambda(lambda x: torch.from_numpy(x)),
                                       trfs.Lambda(
                                           lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x)])

masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                      transformations.NiftiOrderTransformation(),
                                      trfs.Lambda(lambda x: torch.from_numpy(x)),
                                      transformations.BinarizationTransformation(),
                                      trfs.Lambda(
                                          lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 2 != 0 else x)])

train_images_set = datasets.NiftiFolder(train_images_path, images_transformations)
train_masks_set = datasets.NiftiFolder(train_masks_path, masks_transformations)

valid_images_set = datasets.NiftiFolder(valid_masks_path, images_transformations)
valid_mask_set = datasets.NiftiFolder(valid_masks_path, masks_transformations)

train_set = datasets.CombinedDataset(train_images_set, train_masks_set)
valid_set = datasets.CombinedDataset(valid_images_set, valid_mask_set)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, shuffle=True)

net = UNet3D(1, 1).double()
criterion = DiceLossOneClass()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train_runner = TrainingEpochRunner(optimizer, criterion)
valid_runner = ValidationEpochRunner(criterion)

trainer = PyTorchTrainer(net, train_runner, valid_runner)
trainer.add_observer(TensorboardLogger("./logs"))
trainer.add_observer(PyTorchWeightsSaver("./logs"))

trainer.perform_training(10, train_loader, valid_loader)
