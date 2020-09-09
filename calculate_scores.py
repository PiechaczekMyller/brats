import argparse
import enum
import json
import os
import warnings

import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torchvision import transforms as trfs
from brats import transformations
from brats.metrics import DiceScore, RecallScore, PrecisionScore, HausdorffDistance95, FScore
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

    masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                          transformations.NiftiToTorchDimensionsReorderTransformation(),
                                          trfs.Lambda(lambda x: torch.from_numpy(x)),
                                          transformations.OneHotEncoding([0, 1, 2, 3]),
                                          trfs.Lambda(
                                              lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[1] % 16 != 0 else x),
                                          trfs.Lambda(lambda x: x.float()),
                                          transformations.ResizeVolumeTransformation((args.input_size, args.input_size))
                                          ])

    with open(args.division_json) as division_json:
        division = json.load(division_json)

    valid_patients = division["valid"]
    test_patients = division["test"]

    valid_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in valid_patients]
    test_masks_paths = [os.path.join(args.dataset_path, "labelsTr", patient) for patient in test_patients]

    valid_mask_set = datasets.NiftiFolder(valid_masks_paths, masks_transformations)
    test_mask_set = datasets.NiftiFolder(test_masks_paths, masks_transformations)

    valid_mask_loader = torch.utils.data.DataLoader(valid_mask_set, batch_size=args.batch_size, shuffle=False)
    test_mask_loader = torch.utils.data.DataLoader(test_mask_set, batch_size=args.batch_size, shuffle=False)

    dice = DiceScore()
    recall = RecallScore()
    precision = PrecisionScore()
    fscore = FScore(0.5)
    hausdorff = HausdorffDistance95(merge_operation=torch.sum)

    one_hot_encoder = transformations.OneHotEncoding([0, 1, 2, 3])

    valid_outputs_path = os.path.join(args.log_dir, 'outputs', 'valid')
    test_outputs_path = os.path.join(args.log_dir, 'outputs', 'test')

    valid_metrics = []

    for mask, mask_path in zip(valid_mask_loader, valid_mask_set._files):
        prediction_path = os.path.join(valid_outputs_path, os.path.basename(mask_path))
        prediction = torch.load(prediction_path)

        _, max_indeces = prediction.max(1)
        one_hot_prediction = one_hot_encoder(max_indeces).unsqueeze(0)

        one_hot_prediction=one_hot_prediction[:, :, 5: ,: ,:]
        mask=mask[:, :, 5:, : , :]

        patient_metrics = {'name': os.path.basename(mask_path)}

        dice_scores = dice(one_hot_prediction, mask)
        recall_scores = recall(one_hot_prediction, mask)
        precision_scores = precision(one_hot_prediction, mask)
        fscore_scores = fscore(one_hot_prediction, mask)
        # hausdorff_scores = hausdorff(one_hot_prediction, mask)

        wt_prediction = torch.ones_like(one_hot_prediction) - one_hot_prediction[:,0,...]
        wt_prediction = wt_prediction[:, 0 ,...]
        wt_prediction = wt_prediction.unsqueeze(0)

        wt_mask = torch.ones_like(mask) - mask[:, 0, ...]
        wt_mask = wt_mask[:, 0, ...]
        wt_mask = wt_mask.unsqueeze(0)

        patient_metrics['dice_edema'] = dice_scores[Labels.EDEMA].item()
        patient_metrics['dice_enhancing'] = dice_scores[Labels.ENHANCING].item()
        patient_metrics['dice_non_enhancing'] = dice_scores[Labels.NON_ENHANCING].item()
        patient_metrics['dice_whole_tumor'] = dice(wt_prediction, wt_mask)[0].item()

        patient_metrics['recall_edema'] = recall_scores[Labels.EDEMA].item()
        patient_metrics['recall_enhancing'] = recall_scores[Labels.ENHANCING].item()
        patient_metrics['recall_non_enhancing'] = recall_scores[Labels.NON_ENHANCING].item()
        patient_metrics['recall_whole_tumor'] = recall(wt_prediction, wt_mask)[0].item()

        patient_metrics['precision_edema'] = precision_scores[Labels.EDEMA].item()
        patient_metrics['precision_enhancing'] = precision_scores[Labels.ENHANCING].item()
        patient_metrics['precision_non_enhancing'] = precision_scores[Labels.NON_ENHANCING].item()
        patient_metrics['precision_whole_tumor'] = precision(wt_prediction, wt_mask)[0].item()

        patient_metrics['fscore_edema'] = fscore_scores[Labels.EDEMA].item()
        patient_metrics['fscore_enhancing'] = fscore_scores[Labels.ENHANCING].item()
        patient_metrics['fscore_non_enhancing'] = fscore_scores[Labels.NON_ENHANCING].item()
        patient_metrics['fscore_whole_tumor'] = fscore(wt_prediction, wt_mask)[0].item()

        # patient_metrics['hausdorff_edema'] = hausdorff_scores[Labels.EDEMA].item()
        # patient_metrics['hausdorff_enhancing'] = hausdorff_scores[Labels.ENHANCING].item()
        # patient_metrics['hausdorff_non_enhancing'] = hausdorff_scores[Labels.NON_ENHANCING].item()
        # patient_metrics['hausdorff_whole_tumor'] = hausdorff(wt_prediction, wt_mask)[0].item()

        valid_metrics.append(patient_metrics)
    valid_df = pd.DataFrame(valid_metrics)
    valid_df.to_csv(os.path.join(args.log_dir, "valid_scores.csv"), index=False)

    test_metrics = []

    for mask, mask_path in zip(test_mask_loader, test_mask_set._files):
        prediction_path = os.path.join(test_outputs_path, os.path.basename(mask_path))
        prediction = torch.load(prediction_path)

        _, max_indeces = prediction.max(1)
        one_hot_prediction = one_hot_encoder(max_indeces).unsqueeze(0)

        one_hot_prediction = one_hot_prediction[:, :, 5:, :, :]
        mask = mask[:, :, 5:, :, :]

        patient_metrics = {'name': os.path.basename(mask_path)}

        dice_scores = dice(one_hot_prediction, mask)
        recall_scores = recall(one_hot_prediction, mask)
        precision_scores = precision(one_hot_prediction, mask)
        fscore_scores = fscore(one_hot_prediction, mask)
        # hausdorff_scores = hausdorff(one_hot_prediction, mask)

        wt_prediction = torch.ones_like(one_hot_prediction) - one_hot_prediction[:, 0, ...]
        wt_prediction = wt_prediction[:, 0, ...]
        wt_prediction = wt_prediction.unsqueeze(0)

        wt_mask = torch.ones_like(mask) - mask[:, 0, ...]
        wt_mask = wt_mask[:, 0, ...]
        wt_mask = wt_mask.unsqueeze(0)

        patient_metrics['dice_edema'] = dice_scores[Labels.EDEMA].item()
        patient_metrics['dice_enhancing'] = dice_scores[Labels.ENHANCING].item()
        patient_metrics['dice_non_enhancing'] = dice_scores[Labels.NON_ENHANCING].item()
        patient_metrics['dice_whole_tumor'] = dice(wt_prediction, wt_mask)[0].item()

        patient_metrics['recall_edema'] = recall_scores[Labels.EDEMA].item()
        patient_metrics['recall_enhancing'] = recall_scores[Labels.ENHANCING].item()
        patient_metrics['recall_non_enhancing'] = recall_scores[Labels.NON_ENHANCING].item()
        patient_metrics['recall_whole_tumor'] = recall(wt_prediction, wt_mask)[0].item()

        patient_metrics['precision_edema'] = precision_scores[Labels.EDEMA].item()
        patient_metrics['precision_enhancing'] = precision_scores[Labels.ENHANCING].item()
        patient_metrics['precision_non_enhancing'] = precision_scores[Labels.NON_ENHANCING].item()
        patient_metrics['precision_whole_tumor'] = precision(wt_prediction, wt_mask)[0].item()

        patient_metrics['fscore_edema'] = fscore_scores[Labels.EDEMA].item()
        patient_metrics['fscore_enhancing'] = fscore_scores[Labels.ENHANCING].item()
        patient_metrics['fscore_non_enhancing'] = fscore_scores[Labels.NON_ENHANCING].item()
        patient_metrics['fscore_whole_tumor'] = fscore(wt_prediction, wt_mask)[0].item()

        # patient_metrics['hausdorff_edema'] = hausdorff_scores[Labels.EDEMA].item()
        # patient_metrics['hausdorff_enhancing'] = hausdorff_scores[Labels.ENHANCING].item()
        # patient_metrics['hausdorff_non_enhancing'] = hausdorff_scores[Labels.NON_ENHANCING].item()
        # patient_metrics['hausdorff_whole_tumor'] = hausdorff(wt_prediction, wt_mask)[0].item()

        test_metrics.append(patient_metrics)
    test_df = pd.DataFrame(test_metrics)
    test_df.to_csv(os.path.join(args.log_dir, "test_scores.csv"), index=False)
