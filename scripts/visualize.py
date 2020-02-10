import os
import torch
import warnings
import numpy as np
from argparse import ArgumentParser

from torch.nn import functional as F
from torchvision import transforms as trfs
from skimage import io
from skimage.color import label2rgb

from brats import transformations
from brats.models import UNet3D
from brats.data import datasets
import brats.functional as brats_f
from brats.utils import convert_image_to_uint8

CLASS_DIM = 1
DEPTH = 0
FLAIR_MODALITY = 3


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        help='Path to the folder with patients')
    parser.add_argument('--masks_path', type=str,
                        help='Path to the corresponding masks')
    parser.add_argument('--model_weights_path', type=str,
                        help='Path to the model parameters')
    parser.add_argument('--output_dir', type=str,
                        help='Directory for visualizations')
    parser.add_argument('--input_size', nargs='+', type=tuple, default=None,
                        help='Size to which the volumes should be changed.')
    parser.add_argument('--device', type=str, help='CPU or GPU')

    return parser.parse_args()


def get_masks_transformations(input_size, device):
    masks_transformations = trfs.Compose(
        [transformations.AddChannelDimToMaskTransformation(),
         transformations.NiftiToTorchDimensionsReorderTransformation(),
         trfs.Lambda(lambda x: torch.from_numpy(x)),
         transformations.BinarizationTransformation(),
         trfs.Lambda(
             lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[
                                                           1] % 16 != 0 else x),
         transformations.ResizeVolumeTransformation(input_size),
         trfs.Lambda(lambda x: x.float()),
         trfs.Lambda(lambda x: x.to(device))
         ])
    return masks_transformations


def get_volumes_transformations(input_size, device):
    volumes_transformations = trfs.Compose(
        [transformations.NiftiToTorchDimensionsReorderTransformation(),
         trfs.Lambda(lambda x: x[FLAIR_MODALITY, :, :, :]),
         trfs.Lambda(lambda x: np.expand_dims(x, 0)),
         trfs.Lambda(lambda x: torch.from_numpy(x)),
         trfs.Lambda(
             lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[
                                                           1] % 16 != 0 else x),
         transformations.StandardizeVolume(),
         transformations.ResizeVolumeTransformation(input_size),
         trfs.Lambda(lambda x: x.float()),
         trfs.Lambda(lambda x: x.to(device))
         ])
    return volumes_transformations


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    volumes_transformations = get_volumes_transformations((96, 96), args.device)
    masks_transformations = get_masks_transformations((96, 96), args.device)
    valid_images_set = datasets.NiftiFolder(args.dataset_path,
                                            volumes_transformations)
    masks_set = datasets.NiftiFolder(args.masks_path, masks_transformations)
    combined_set = datasets.CombinedDataset(valid_images_set, masks_set)
    net = UNet3D(1, 1).float()
    net = net.to(args.device)
    net.load_state_dict(torch.load(args.model_weights_path,
                                   map_location=torch.device(args.device)))
    net.eval()
    dataloader = torch.utils.data.DataLoader(combined_set, batch_size=1,
                                             shuffle=False)
    for idx, (volume, mask) in enumerate(dataloader):
        folder = 'patient_{}'.format(idx)
        patient_path = os.path.join(args.output_dir, folder)
        os.makedirs(patient_path, exist_ok=True)
        with torch.no_grad():
            output = net(volume)
        dice_score = brats_f.dice(output, mask)
        np.savetxt(os.path.join(patient_path, "dice_score.txt"),
                   dice_score.numpy().reshape(1))
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        # output = torch.argmax(output, dim=CLASS_DIM)[0, ...]
        output = output[0, 0, ...]
        # mask = torch.argmax(mask, dim=CLASS_DIM)[0, ...]
        mask = mask[0, 0, ...]
        mask[mask == 0] = -1
        output[output == 0] = -1
        output = output.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        volume = volume.detach().cpu().numpy()
        volume = volume[0, 0, ...]
        for slice_idx in range(output.shape[DEPTH]):
            slice_rgb = label2rgb(output[slice_idx], image=volume[slice_idx],
                                  image_alpha=0.2)
            mask_rgb = label2rgb(mask[slice_idx], image=volume[slice_idx],
                                 image_alpha=0.2)
            slice_rgb = convert_image_to_uint8(slice_rgb)
            mask_rgb = convert_image_to_uint8(mask_rgb)
            filename_slice = '{}_slice.png'.format(slice_idx)
            filename_mask = '{}_mask.png'.format(slice_idx)
            io.imsave(os.path.join(patient_path, filename_slice), slice_rgb)
            io.imsave(os.path.join(patient_path, filename_mask), mask_rgb)
