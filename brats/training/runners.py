import numpy as np
import torch
from torch import nn
from torch.utils import data


def run_training_epoch(model: nn.Module, data_loader: data.DataLoader, optimizer, criterion, metrics, device):
    """
    Function performing one training epoch.
    Args:
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
            out_metrics[metric_name].append(metrics[metric_name](output, target).item())
    out_metrics = {metric_name: np.mean(out_metrics[metric_name]) for metric_name in metrics.keys()}
    return (np.mean(losses), out_metrics) if losses else (0.0, out_metrics)


def run_validation_epoch(model: nn.Module, data_loader: data.DataLoader, criterion, metrics, device):
    """
    Function performing one validation epoch.
    Args:
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
                out_metrics[metric_name].append(metrics[metric_name](output, target).item())
        out_metrics = {metric_name: np.mean(out_metrics[metric_name]) for metric_name in metrics.keys()}
    return (np.mean(losses), out_metrics) if losses else (0.0, out_metrics)
