"""
Observers - object that wait for particular event and somehow response to them.
Most of observers that can be found in this module are use to log information about traning process.

"""
import abc
import os
import csv
import time
import collections
import statistics
import tensorboardX
import torch

from typing import Any, List, Union

Path = Union[os.PathLike, str]


class Observer(abc.ABC):
    @abc.abstractmethod
    def on_event(self, event, data: Any):
        """
        Callback which is called every time observed object fires an event.

        :param event: Event to handle
        :param data: Data associated with event
        """


class Observable:
    """
    Objects that are observable can notify different observers of specific
    events.

    """

    def __init__(self):
        self.observers = []

    def add_observer(self, observer: Observer):
        """
        Adds observer so it can be notified of events.

        """
        self.observers.append(observer)

    def notify(self, event, data: Any):
        for observer in self.observers:
            observer.on_event(event, data)


class TensorboardLogger(Observer):
    """
    Writes tensorboard file with some training statistics

    """
    LogPoint = collections.namedtuple("LogPoint", ["name", "value", "epoch"])

    def __init__(self, log_directory: Path):
        tensorboard_path = os.path.join(log_directory, "tensorlog")
        self.writer = tensorboardX.SummaryWriter(tensorboard_path)

    def on_event(self, event, data: List[LogPoint]):
        considered_events = ["training_epoch_finished",
                             "validation_epoch_finished"]
        if event in considered_events:
            for log_point in data:
                self.writer.add_scalar(log_point.name, log_point.value, log_point.epoch)


AfterEpochDataToSave = collections.namedtuple("DataToSave", ["model", "state_dict"])


class PyTorchModelSaver(Observer):
    MODEL_FILENAME = "model"

    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self._epoch = 0

    def on_event(self, event, data: AfterEpochDataToSave):
        if event == "epoch_finished":
            model_path = os.path.join(self.log_directory, self.MODEL_FILENAME + f"_{str(self._epoch).zfill(6)}")
            self._epoch += 1
            with open(model_path, "wb") as file:
                torch.save(data.model, file)


class FirstValidationBatchSaver(Observer):
    EpochValidationResults = collections.namedtuple("EpochValidationResults",
                                                    ["epoch", "data"])

    class ResultsSaver:
        def save(self, filename_prefix: str, data: torch.Tensor):
            pass

    SUBDIR_NAME = "validation"

    def __init__(self, log_directory: Path, results_saver: ResultsSaver):
        self.log_directory = log_directory
        self.processed_epochs = set()
        self.results_saver = results_saver
        os.makedirs(os.path.join(self.log_directory, self.SUBDIR_NAME), exist_ok=True)

    def _should_run(self, epoch):
        return epoch not in self.processed_epochs

    def on_event(self, event, data: EpochValidationResults):
        if event != "validation_batch_output":
            return
        if self._should_run(data.epoch):
            self.processed_epochs.add(data.epoch)
            output_prefix = os.path.join(self.log_directory, self.SUBDIR_NAME, str(data.epoch).zfill(6))
            self.results_saver.save(output_prefix, data.data)


class PyTorchWeightsSaver(Observer):
    WEIGHTS_FILENAME = "weights"

    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self._epoch = 0

    def on_event(self, event, data: AfterEpochDataToSave):
        if event == "epoch_finished":
            weights_path = os.path.join(self.log_directory, self.WEIGHTS_FILENAME + f"_{str(self._epoch).zfill(6)}")
            self._epoch += 1
            with open(weights_path, "wb") as file:
                torch.save(data.state_dict, file)


class KerasModelSaver(Observer):
    DataToSave = collections.namedtuple("DataToSave",
                                        ["file_name", "model"])

    def __init__(self, log_directory: Path):
        self.log_directory = log_directory

    def on_event(self, event, data: DataToSave):
        if event == "epoch_finished":
            for data_point in data:
                path = os.path.join(self.log_directory, data_point.file_name)
                data_point.model.save(path)


class LossLogger(Observer):
    LOSS_LOG_FILENAME = "losses.csv"

    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self.log_path = os.path.join(self.log_directory, self.LOSS_LOG_FILENAME)
        self._last_train_loss = None
        self._last_validation_loss = None
        self._last_epoch = None

    def on_event(self, event, data):
        if event == "training_began":
            with open(self.log_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss", "val_loss"])
        if event == "training_epoch_finished":
            for log_point in data:
                if log_point.name == "training_loss":
                    self._last_train_loss = log_point.value
                    self._last_epoch = log_point.epoch
        if event == "validation_epoch_finished":
            for log_point in data:
                if log_point.name == "validation_loss":
                    self._last_validation_loss = log_point.value
                    self._last_epoch = log_point.epoch
        if event == "epoch_finished":
            with open(self.log_path, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([self._last_epoch, f"{self._last_train_loss:.6f}", f"{self._last_validation_loss:.6f}"])
                print(f"{self._last_epoch}: {self._last_train_loss:.6f}, {self._last_validation_loss:.6f}")


class TimeLogger(Observer):
    TIME_LOG_FILENAME = "times.csv"

    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self._training_started = None
        self._epoch_times_since_beginning = []
        self._avgs = []

    def on_event(self, event, data):
        log_path = os.path.join(self.log_directory, self.TIME_LOG_FILENAME)
        if event == "training_began":
            self._training_started = time.time()
            with open(log_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "time", "total_time", "average_time"])
        if event == "epoch_finished":
            self._epoch_times_since_beginning.append(time.time() - self._training_started)
            with open(log_path, "a", newline="") as file:
                writer = csv.writer(file)
                current_epoch = len(self._epoch_times_since_beginning)
                epoch_time = (self._epoch_times_since_beginning[0] if len(self._epoch_times_since_beginning) == 1
                              else (self._epoch_times_since_beginning[current_epoch - 1] -
                                    self._epoch_times_since_beginning[current_epoch - 2]))
                self._avgs.append(epoch_time)
                current_mean = statistics.mean(self._avgs)
                current_total = self._epoch_times_since_beginning[-1]
                writer.writerow([current_epoch, f"{epoch_time:.4f}", f"{current_total:.4f}", f"{current_mean:.4f}"])
