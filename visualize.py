import argparse
import enum
import json
import os
import pathlib
import warnings

import cv2
from matplotlib import pyplot as plt

from skimage import io, color
import torch
import numpy as np
from skimage.color import label2rgb
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.data import datasets

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


class PadTo160:
    def __call__(self, x):
        a = F.pad(x, [0, 0, 0, 0, 16 - (x.shape[1] % 16), 0]) if x.shape[1] % 16 != 0 else x
        return a


def add_red_label(image, label):
    if image.ndim == 2:
        image = color.gray2rgb(image)
    mask = np.stack([label, np.zeros_like(label), np.zeros_like(label)], axis=2)
    image = image.astype(np.float64)
    mask = mask.astype(np.float64)
    masked = cv2.addWeighted(image, 1., mask, 0.5, 0)
    return masked


def add_green_label(image, label):
    if image.ndim == 2:
        image = color.gray2rgb(image)
    mask = np.stack([np.zeros_like(label), label, np.zeros_like(label)], axis=2)
    image = image.astype(np.float64)
    mask = mask.astype(np.float64)
    masked = cv2.addWeighted(image, 1., mask, 0.5, 0)
    return masked


def add_yellow_label(image, label):
    if image.ndim == 2:
        image = color.gray2rgb(image)
    mask = np.stack([label, label, np.zeros_like(label)], axis=2)
    image = image.astype(np.float64)
    mask = mask.astype(np.float64)
    masked = cv2.addWeighted(image, 1., mask, 0.5, 0)
    return masked


def add_cyan_label(image, label):
    if image.ndim == 2:
        image = color.gray2rgb(image)
    mask = np.stack([np.zeros_like(label), label, label], axis=2)
    image = image.astype(np.float64)
    mask = mask.astype(np.float64)
    masked = cv2.addWeighted(image, 1., mask, 0.5, 0)
    return masked


errors_colors = {1: add_green_label, 2: add_yellow_label, 3: add_red_label}
classes_colors = {1: add_cyan_label, 2: add_yellow_label, 3: add_red_label}


def draw_labels(labels, image, colors):
    image = convert_image_to_uint8(image).astype(np.float64) / 255
    masked = np.copy(image)
    for class_id in np.unique(labels):
        if class_id == 0:
            continue
        mask = np.zeros_like(image)
        mask[labels == class_id] = 1
        masked = colors[class_id](masked, mask)
    return masked


def create_error_vis_for_classes(image, prediction, mask):
    images = {}
    for tumor_class in Labels:
        mask_for_class = np.zeros_like(mask)
        mask_for_class[mask == tumor_class] = 1

        mask_for_pred = np.zeros_like(prediction)
        mask_for_pred[prediction == tumor_class] = 1

        errors = mask_for_class - 2 * mask_for_pred

        error_labels = np.zeros_like(mask)
        error_labels[errors == -1] = 1  # True positives
        error_labels[errors == -2] = 2  # False positives
        error_labels[errors == 1] = 3  # False negatives

        class_error_image = draw_labels(error_labels,
                                        image,
                                        colors=errors_colors)
        images[tumor_class] = class_error_image
    return images


def create_wt_image(image, prediction, mask):
    wt_mask = np.zeros_like(mask)
    wt_mask[mask > 0] = 1
    wt_pred = np.zeros_like(prediction)
    wt_pred[prediction > 0] = 1

    errors = wt_mask - 2 * wt_pred

    error_labels = np.zeros_like(mask)
    error_labels[errors == -1] = 1  # True positives
    error_labels[errors == -2] = 2  # False positives
    error_labels[errors == 1] = 3  # False negatives

    class_error_image = draw_labels(error_labels,
                                    image,
                                    colors=errors_colors)
    return class_error_image


def convert_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert the image tu uint8
    Args:
        image: Image to convert
    Returns:
        np.ndarray: Image in the uint8 format
    """
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = 255 * image
    return image.astype(np.uint8)


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

    center_crop_start = (240 - args.input_size) // 2
    center_crop_stop = (240 - args.input_size) // 2 + args.input_size
    volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                            trfs.Lambda(lambda x: torch.from_numpy(x)),
                                            PadTo160(),
                                            trfs.Lambda(lambda x: x.float()),
                                            transformations.ResizeVolumeTransformation(
                                                (args.input_size, args.input_size))
                                            ])
    masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          PadTo160(),
                                          trfs.Lambda(lambda x: x.float()),
                                          transformations.ResizeVolumeTransformation((args.input_size, args.input_size))
                                          ])

    with open(args.division_json) as division_json:
        division = json.load(division_json)

    valid_patients = division["valid"]
    test_patients = division["test"]

    valid_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in valid_patients]
    valid_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in valid_patients]
    test_volumes_paths = [os.path.join(args.dataset_path, "imagesTr", patient) for patient in test_patients]
    test_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in test_patients]

    valid_volumes_set = datasets.NiftiFolder(valid_volumes_paths, volumes_transformations)
    valid_mask_set = datasets.NiftiFolder(valid_masks_paths, masks_transformations)
    test_volumes_set = datasets.NiftiFolder(test_volumes_paths, volumes_transformations)
    test_mask_set = datasets.NiftiFolder(test_masks_paths, masks_transformations)

    valid_volumes_loader = torch.utils.data.DataLoader(valid_volumes_set, batch_size=args.batch_size, shuffle=False)
    valid_mask_loader = torch.utils.data.DataLoader(valid_mask_set, batch_size=args.batch_size, shuffle=False)
    test_volumes_loader = torch.utils.data.DataLoader(test_volumes_set, batch_size=args.batch_size, shuffle=False)
    test_mask_loader = torch.utils.data.DataLoader(test_mask_set, batch_size=args.batch_size, shuffle=False)

    one_hot_encoder = transformations.OneHotEncoding([0, 1, 2])

    valid_outputs_path = os.path.join(args.log_dir, 'outputs', 'valid')
    test_outputs_path = os.path.join(args.log_dir, 'outputs', 'test')

    for volume, mask, patient_path in zip(valid_volumes_set, valid_mask_set, valid_volumes_set._files):
        prediction_path = os.path.join(valid_outputs_path, os.path.basename(patient_path))
        prediction = torch.load(prediction_path)

        _, max_indeces = prediction.max(1)

        volume = volume[0, ...].cpu().numpy()
        max_indeces = max_indeces[0, ...].cpu().numpy()
        mask = mask[0, ...].cpu().numpy()

        output_dir = os.path.join(args.log_dir, "visualization", 'valid')
        for slice_idx in range(max_indeces.shape[0]):
            prediction_rgb = draw_labels(max_indeces[slice_idx],
                                         image=volume[slice_idx, ...],
                                         colors=classes_colors)
            mask_rgb = draw_labels(mask[slice_idx, ...],
                                   image=volume[slice_idx, ...],
                                   colors=classes_colors)
            class_images = create_error_vis_for_classes(volume[slice_idx, ...],
                                                        max_indeces[slice_idx, ...],
                                                        mask[slice_idx, ...])
            whole_tumor_image = create_wt_image(volume[slice_idx, ...],
                                                max_indeces[slice_idx, ...],
                                                mask[slice_idx, ...])

            prediction_rgb = convert_image_to_uint8(prediction_rgb)
            mask_rgb = convert_image_to_uint8(mask_rgb)
            wt_rgb = convert_image_to_uint8(whole_tumor_image)
            class_images = {key: convert_image_to_uint8(image) for key, image in class_images.items()}

            slice_filename = '{}.png'.format(slice_idx)
            prediction_path = os.path.join(output_dir, os.path.basename(patient_path), 'prediction',
                                           slice_filename)
            mask_path = os.path.join(output_dir, os.path.basename(patient_path), 'mask', slice_filename)
            background_path = os.path.join(output_dir, os.path.basename(patient_path), 'background', slice_filename)
            edema_path = os.path.join(output_dir, os.path.basename(patient_path), 'edema', slice_filename)
            non_enh_path = os.path.join(output_dir, os.path.basename(patient_path), 'non_enhancing', slice_filename)
            enh_path = os.path.join(output_dir, os.path.basename(patient_path), 'enhancing', slice_filename)
            wt_path = os.path.join(output_dir, os.path.basename(patient_path), 'whole_tumor', slice_filename)

            pathlib.Path(os.path.dirname(prediction_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(mask_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(background_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(edema_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(non_enh_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(enh_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(wt_path)).mkdir(parents=True, exist_ok=True)

            io.imsave(prediction_path, prediction_rgb)
            io.imsave(mask_path, mask_rgb)
            io.imsave(background_path, class_images[Labels.BACKGROUND])
            io.imsave(edema_path, class_images[Labels.EDEMA])
            io.imsave(non_enh_path, class_images[Labels.NON_ENHANCING])
            # io.imsave(enh_path, class_images[Labels.ENHANCING])
            io.imsave(wt_path, wt_rgb)

    for volume, mask, patient_path in zip(test_volumes_set, test_mask_set, test_volumes_set._files):
        prediction_path = os.path.join(test_outputs_path, os.path.basename(patient_path))
        prediction = torch.load(prediction_path)

        _, max_indeces = prediction.max(1)

        volume = volume[0, ...].cpu().numpy()
        max_indeces = max_indeces[0, ...].cpu().numpy()
        mask = mask[0, ...].cpu().numpy()

        output_dir = os.path.join(args.log_dir, "visualization", 'test')
        for slice_idx in range(max_indeces.shape[0]):
            prediction_rgb = draw_labels(max_indeces[slice_idx],
                                         image=volume[slice_idx, ...],
                                         colors=classes_colors)
            mask_rgb = draw_labels(mask[slice_idx, ...],
                                   image=volume[slice_idx, ...],
                                   colors=classes_colors)
            class_images = create_error_vis_for_classes(volume[slice_idx, ...],
                                                        max_indeces[slice_idx, ...],
                                                        mask[slice_idx, ...])
            whole_tumor_image = create_wt_image(volume[slice_idx, ...],
                                                max_indeces[slice_idx, ...],
                                                mask[slice_idx, ...])

            prediction_rgb = convert_image_to_uint8(prediction_rgb)
            mask_rgb = convert_image_to_uint8(mask_rgb)
            wt_rgb = convert_image_to_uint8(whole_tumor_image)
            class_images = {key: convert_image_to_uint8(image) for key, image in class_images.items()}

            slice_filename = '{}.png'.format(slice_idx)
            prediction_path = os.path.join(output_dir, os.path.basename(patient_path), 'prediction',
                                           slice_filename)
            mask_path = os.path.join(output_dir, os.path.basename(patient_path), 'mask', slice_filename)
            background_path = os.path.join(output_dir, os.path.basename(patient_path), 'background', slice_filename)
            edema_path = os.path.join(output_dir, os.path.basename(patient_path), 'edema', slice_filename)
            non_enh_path = os.path.join(output_dir, os.path.basename(patient_path), 'non_enhancing', slice_filename)
            enh_path = os.path.join(output_dir, os.path.basename(patient_path), 'enhancing', slice_filename)
            wt_path = os.path.join(output_dir, os.path.basename(patient_path), 'whole_tumor', slice_filename)

            pathlib.Path(os.path.dirname(prediction_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(mask_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(background_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(edema_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(non_enh_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(enh_path)).mkdir(parents=True, exist_ok=True)
            pathlib.Path(os.path.dirname(wt_path)).mkdir(parents=True, exist_ok=True)

            io.imsave(prediction_path, prediction_rgb)
            io.imsave(mask_path, mask_rgb)
            io.imsave(background_path, class_images[Labels.BACKGROUND])
            io.imsave(edema_path, class_images[Labels.EDEMA])
            io.imsave(non_enh_path, class_images[Labels.NON_ENHANCING])
            # io.imsave(enh_path, class_images[Labels.ENHANCING])
            io.imsave(wt_path, wt_rgb)
