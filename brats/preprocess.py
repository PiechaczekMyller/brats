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
    for dir_entry in list(os.scandir(args.dataset_path)):
        image = nib.load(dir_entry.path)
        image_data = image.get_fdata()
        transformed = transforms(image_data)
        transformed = nib.Nifti1Image(transformed, image.affine)
        nib.save(transformed, os.path.join(args.output_dir, dir_entry.name))


if __name__ == '__main__':
    main()
