import typing

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils import data


def run_training_epoch(model: nn.Module, data_loader: data.DataLoader, optimizer: Optimizer,
                       criterion: typing.Callable,
                       metrics: typing.Dict[str, typing.Callable], device: str):
    """
    Function performing one training epoch.
    Args:
        :param model: Network on which epoch is performed.
        :param data_loader: Loader providing data to train from.
        :param optimizer: Optimizer which performs optimization of the training loss.
        :param criterion: Function to calculate training loss.
        :param metrics: Dict containing name of the metric and the function to calculate it.
        :param device: Device where the data will be send.
    """
    model.train(True)
    losses = []
    out_metrics = {metric_name: [] for metric_name in metrics.keys()}
    for input, target in data_loader:
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        for metric_name in metrics:
            metric_values = [value.item() for value in metrics[metric_name](output, target)]
            out_metrics[metric_name].append(metric_values)
    out_metrics = {metric_name: np.array(out_metrics[metric_name]).mean(axis=1) for metric_name in metrics.keys()}
    return (np.mean(losses), out_metrics) if losses else (0.0, out_metrics)


def run_validation_epoch(model: nn.Module, data_loader: data.DataLoader, criterion: typing.Callable,
                         metrics: typing.Dict[str, typing.Callable], device: str):
    """
    Function performing one validation epoch.
    Args:
        :param device: Device where the data will be send.
        :param model: Network on which epoch is performed.
        :param data_loader: Loader providing data on which model is validated.
        :param criterion: Function to calculate validation loss.
        :param metrics: Dict containing name of the metric and the function to calculate it.
        :param device: Device where the data will be send.
    """
    model.train(False)
    losses = []
    out_metrics = {metric_name: [] for metric_name in metrics.keys()}
    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())
            for metric_name in metrics:
                metric_values = [value.item() for value in metrics[metric_name](output, target)]
                out_metrics[metric_name].append(metric_values)
        out_metrics = {metric_name: np.array(out_metrics[metric_name]).mean(axis=1) for metric_name in metrics.keys()}
    return (np.mean(losses), out_metrics) if losses else (0.0, out_metrics)
