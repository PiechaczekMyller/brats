import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.losses import DiceLossOneClass
from brats.models import UNet3D
from brats.data import datasets

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

net = UNet3D(1, 1).float()
criterion = DiceLossOneClass()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
