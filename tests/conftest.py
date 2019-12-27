import numpy as np
import torch
from torch.utils import data
from torch import nn

INPUT_IMAGE_SHAPE = (33, 33)
INPUT_IMAGES = 2


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(1, 1))

    def forward(self, x):
        return x


class EmptyDataset(data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        raise RuntimeError("getitem called on empty dataset.")


class DummyDataset(data.Dataset):
    def __init__(self, items_in_dataset: int):
        self._items = [np.ones(INPUT_IMAGE_SHAPE, dtype=np.uint8) for _ in
                       range(items_in_dataset)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return self._items[item]


class TensorInputLabelDataset(data.Dataset):
    def __init__(self):
        self.inputs = list(torch.ones(1, 1))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return self.inputs[item], self.inputs[item]


class OneToOneModel(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x
