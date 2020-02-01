import collections
import json
import os
import pathlib

import torch
from typing import Union

import tensorboardX
from torch import nn


class TensorboardLogger:
    """
    Writes tensorboard file with some training statistics

    """

    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(log_directory, "tensorboard")
        self.writer = tensorboardX.SummaryWriter(tensorboard_path)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class StateDictsLogger:
    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory

    def log(self, model: nn.Module, epoch):
        weights_path = os.path.join(self.log_directory, f"state_dict_{str(epoch).zfill(6)}")
        with open(weights_path, "wb") as file:
            torch.save(model.state_dict, file)


class ModelLogger:
    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory

    def log(self, model: nn.Module, epoch):
        model_path = os.path.join(self.log_directory, f"model_{str(epoch).zfill(6)}")
        with open(model_path, "wb") as file:
            torch.save(model, file)


class BestModelLogger:
    def __init__(self, log_directory):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory
        self.best_score = 0

    def log(self, model: nn.Module, score: float):
        if score > self.best_score:
            self.best_score = score
            model_path = os.path.join(self.log_directory, f"best_model")
            with open(model_path, "wb") as file:
                torch.save(model, file)


class BestStateDictLogger:
    def __init__(self, log_directory):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory
        self.best_score = 0

    def log(self, model: nn.Module, score: float):
        if score > self.best_score:
            self.best_score = score
            model_path = os.path.join(self.log_directory, f"best_state_dict")
            with open(model_path, "wb") as file:
                torch.save(model, file)


def log_parameters(log_directory, parameters):
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_directory, "parameters.json"), 'w') as fp:
        json.dump(vars(parameters), fp)
