import numpy as np
import torch
from torch import nn
from torch.utils import data


def run_training_epoch(model: nn.Module, data_loader: data.DataLoader, optimizer, criterion, device):
    model.train(True)
    losses = []
    for input, target in data_loader:
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses) if losses else 0.0


def run_validation_epoch(model: nn.Module, data_loader: data.DataLoader, criterion, device):
    model.train(False)
    losses = []
    with torch.no_grad():
        for input, target in data_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())
    return np.mean(losses) if losses else 0.0
