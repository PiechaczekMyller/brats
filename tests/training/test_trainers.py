"""
Tests for trainers.py
"""
from unittest import mock
import pytest
from torch.utils import data
from brats.training import trainers
from ..conftest import TensorInputLabelDataset, SimpleModel, EmptyDataset


class CriterionMock:
    def __init__(self):
        self.got_called = False

    def __call__(self, input, target):
        class LossMock:
            def backward(self):
                pass

            def __float__(self):
                return 0.0

        return LossMock()


def make_new_loader() -> data.DataLoader:
    return data.DataLoader(TensorInputLabelDataset())


def make_empty_loader():
    return data.DataLoader(EmptyDataset())


class TestPyTorchTrainer:
    TEST_EPOCHS = 2
    EPOCH_EVENTS = 3
    ONE_TIME_EVENTS = 1

    @pytest.fixture()
    def trainer(self):
        model = SimpleModel()
        trainer = trainers.PyTorchTrainer(model,
                                          mock.Mock(),
                                          mock.Mock()
                                          )
        return trainer

    def test_perform_training_runs_training_and_validation_runners(self, trainer):
        # when
        trainer.perform_training(self.TEST_EPOCHS, make_new_loader(), make_empty_loader())

        # then
        assert trainer.training_runner.run_epoch.call_count == self.TEST_EPOCHS
        assert trainer.validation_runner.run_epoch.call_count == self.TEST_EPOCHS

    def test_perform_training_sends_notifications(self, trainer):
        # given
        trainer.notify = mock.Mock()

        # when
        trainer.perform_training(self.TEST_EPOCHS, make_new_loader(), make_empty_loader())

        # then
        assert trainer.notify.call_count == self.TEST_EPOCHS * self.EPOCH_EVENTS + self.ONE_TIME_EVENTS
