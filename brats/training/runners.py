import typing
import warnings

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils import data
from torch.cuda.amp import autocast


def run_training_epoch(model: nn.Module, data_loader: data.DataLoader, optimizer: Optimizer, scaler,
                       criterion: typing.Callable,
                       metrics: typing.Dict[str, typing.Callable], device: str, use_amp: bool = False):
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
        with autocast():
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        for metric_name in metrics:
            metric_values = [value.item() for value in metrics[metric_name](output, target)]
            out_metrics[metric_name].append(metric_values)
    out_metrics = {metric_name: np.array(out_metrics[metric_name]).mean(axis=0) for metric_name in metrics.keys()}
    return np.mean(losses), out_metrics


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
            with autocast():
                output = model(input)
                loss = criterion(output, target)
            losses.append(loss.item())
            for metric_name in metrics:
                metric_values = [value.item() for value in metrics[metric_name](output, target)]
                out_metrics[metric_name].append(metric_values)
    out_metrics = {metric_name: np.array(out_metrics[metric_name]).mean(axis=0) for metric_name in metrics.keys()}
    return np.mean(losses), out_metrics


def run_inference(
        model: nn.Module,
        data_loader: data.DataLoader,
        device: str,
):
    """
    Function performing one validation epoch.
    Args:
        :param device: Device where the data will be send.
        :param model: Network on which epoch is performed.
        :param data_loader: Loader providing data on which model is validated.
        :param criterion: Function to calculate validation loss.
        :param device: Device where the data will be send.
    """
    model.train(False)
    model.eval()
    outputs = []
    with torch.no_grad():
        for input in data_loader:
            input = input[0].to(device)
            output = model(input)
            outputs.append(output.detach().cpu())
    return outputs
