import os

import nibabel as nib
from argparse import ArgumentParser
from torchvision.transforms import Compose
import numpy as np
from brats.transformations import HistogramMatchingTransformation


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
    template = np.expand_dims(template,3)
    transforms = Compose([HistogramMatchingTransformation(template)])
    for dir_entry in os.scandir(args.dataset_path):
        image = nib.load(dir_entry.path)
        image_data = image.get_fdata()
        image_data = np.expand_dims(image_data, 3)
        transformed = transforms(image_data)
        transformed = nib.Nifti1Image(transformed, image.affine)
        nib.save(transformed, os.path.join(args.output_dir, dir_entry.name))


if __name__ == '__main__':
    main()
