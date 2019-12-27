"""
Individual runners i.e. classes that execute epochs in a training process.

"""
import array
import math
import statistics
import abc
import torch

from torch.utils import data
from torch import optim
from torch import nn
from brats.utilities import torch_utils, observers

Criterion = nn.Module


class EpochRunner(abc.ABC):

    @abc.abstractmethod
    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int) -> float:
        """


        :param model: Model to run epoch on.
        :param data_loader: Loader return batches of tuples (input, label)
        :return: Epoch loss
        """


class ProgressBar:
    def __init__(self, total: int, prefix: str):
        self._total = total
        self._iter = 0
        self._prefix = prefix
        self.tick()

    def tick(self):
        length = 100
        decimals = 1
        percent = ("{0:." + str(decimals) + "f}").format(100 * (self._iter / float(self._total)))
        filledLength = int(length * self._iter // self._total)
        bar = 'X' * filledLength + '_' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (self._prefix, bar, percent, "Completed"), end='', flush=True)
        self._iter = self._iter + 1

    def finish(self):
        print()


class TrainingEpochRunner(EpochRunner):
    def __init__(self,
                 optimizer: optim.Optimizer,
                 criterion: Criterion,
                 device: str = torch_utils.get_default_processing_device()):
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device

    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int):
        model.train(True)
        batch_losses = array.array('d', [])
        pb = ProgressBar(math.ceil(float(len(data_loader.dataset)) / data_loader.batch_size), f"T({str(epoch).zfill(4)}):")
        for input, target in data_loader:
            input = input.to(self._device)
            target = target.to(self._device)
            self._optimizer.zero_grad()
            output = model(input)
            loss = self._criterion(output, target)
            loss.backward()
            self._optimizer.step()
            batch_losses.append(loss)
            pb.tick()
        pb.finish()
        return statistics.mean(batch_losses) if batch_losses else 0.0


class ValidationEpochRunner(EpochRunner, observers.Observable):
    def __init__(self,
                 criterion: Criterion,
                 device: str = torch_utils.get_default_processing_device()):
        observers.Observable.__init__(self)
        self._criterion = criterion
        self._device = device

    def run_epoch(self, model: nn.Module, data_loader: data.DataLoader, epoch: int):
        model.train(False)
        batch_losses = array.array('d', [])
        pb = ProgressBar(math.ceil(float(len(data_loader.dataset)) / data_loader.batch_size), f"V({str(epoch).zfill(4)}):")
        with torch.no_grad():
            for input, target in data_loader:
                input = input.to(self._device)
                target = target.to(self._device)
                output = model(input)
                self.notify("validation_batch_output",
                            observers.FirstValidationBatchSaver.EpochValidationResults(epoch, output))
                loss = self._criterion(output, target)
                batch_losses.append(loss)
                pb.tick()
        pb.finish()
        return statistics.mean(batch_losses) if batch_losses else 0.0
