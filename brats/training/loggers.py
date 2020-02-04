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
    Logger that creates tensorboard an logs given scalars.
    Args:
        :param log_directory directory where the tensorboard will be created
    """

    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(log_directory, "tensorboard")
        self.writer = tensorboardX.SummaryWriter(tensorboard_path)

    def log(self, name, value, epoch):
        self.writer.add_scalar(name, value, epoch)


class StateDictsLogger:
    """
    Logger that saves states dicts after each epoch.
    Args:
        :param log_directory directory where state dicts will be saved.
    """

    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory

    def log(self, model: nn.Module, epoch):
        weights_path = os.path.join(self.log_directory, f"state_dict_{str(epoch).zfill(6)}")
        with open(weights_path, "wb") as file:
            torch.save(model.state_dict, file)


class ModelLogger:
    """
    Logger that saves models after each epoch.
    Args:
        :param log_directory directory where models will be saved.
    """

    def __init__(self, log_directory: Union[os.PathLike, str]):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory

    def log(self, model: nn.Module, epoch):
        model_path = os.path.join(self.log_directory, f"model_{str(epoch).zfill(6)}")
        with open(model_path, "wb") as file:
            torch.save(model, file)


class BestModelLogger:
    """
    Logger that saves models if they improve best score of the model.
    Args:
        :param log_directory directory where models will be saved.
    """

    def __init__(self, log_directory):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory
        self.best_score = 0

    def log(self, model: nn.Module, score: float):
        """
        Logging method
        Args:
            :param score: Metric, if it is improved, the model is saved (The higher the better).
        """
        if score > self.best_score:
            self.best_score = score
            model_path = os.path.join(self.log_directory, f"best_model")
            with open(model_path, "wb") as file:
                torch.save(model, file)


class BestStateDictLogger:
    """
    Logger that saves state dicts if they improve best score of the model.
    Args:
        :param log_directory directory where state dicts will be saved.
    """

    def __init__(self, log_directory):
        pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
        self.log_directory = log_directory
        self.best_score = 0

    def log(self, model: nn.Module, score: float):
        """
        Logging method
        Args:
            :param score: Metric, if it is improved, the state dict is saved (The higher the better).
        """
        if score > self.best_score:
            self.best_score = score
            model_path = os.path.join(self.log_directory, f"best_state_dict")
            with open(model_path, "wb") as file:
                torch.save(model, file)


def log_parameters(log_directory, parameters):
    """
    Function that saves parameters of the run to json
    Args:
        :param log_directory: Directory where parameters will be saved
        :param parameters: Dict with parameters of the run
    """
    pathlib.Path(log_directory).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_directory, "parameters.json"), 'w') as fp:
        json.dump(vars(parameters), fp)
