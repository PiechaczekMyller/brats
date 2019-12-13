import os

import nibabel as nib
from argparse import ArgumentParser
from torchvision.transforms import Compose

from brats.transformations import HistogramMatchingTransform


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to the folder '
                                                         'with patients')
    parser.add_argument('--template_path', type=str, help='Path to the patient'
                                                          'which will be used'
                                                          'as a template for '
                                                          'histogram matching')
    parser.add_argument('--output_dir', type=str, help='Directory for '
                                                       'preprocessed data')
    return parser.parse_args()


def main():
    args = parse_args()
    template = nib.load(args.template_path)
    template = template.get_fdata()
    transforms = Compose([HistogramMatchingTransform(template)])
    for file_path in list(os.scandir(args.dataset_path)):
        file = nib.load(file_path)
        file = file.get_fdata()
        transformed = transforms(file)
        file_name = os.path.basename(file_path)
        nib.save(transformed, os.path.join(args.output_dir, file_name))
