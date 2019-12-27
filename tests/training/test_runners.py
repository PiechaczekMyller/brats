from unittest import mock
import pytest
from torch.utils import data
from brats.training import runners
from ..conftest import TensorInputLabelDataset, SimpleModel


@pytest.fixture()
def model():
    model = SimpleModel()
    model.train = mock.Mock()
    return model


class TestTrainingEpochRunner:
    @pytest.fixture
    def loss_value(self):
        return 5.0

    @pytest.fixture(scope="class")
    def dataloader(self):
        return data.DataLoader(TensorInputLabelDataset())

    @pytest.fixture
    def runner(self, loss_value):
        optimizer = mock.Mock()
        loss_mock = mock.Mock()
        loss_mock.__float__ = mock.Mock(return_value=loss_value)
        criterion = mock.Mock(return_value=loss_mock)
        runner = runners.TrainingEpochRunner(optimizer, criterion)
        return runner

    def test_run_epoch_turns_on_train_mode(self, runner, model, dataloader):
        # when
        runner.run_epoch(model, dataloader, 1)

        # then
        model.train.assert_called_once_with(True)

    def test_run_epoch_calls_zero_grad(self, runner, model, dataloader):
        # when
        runner.run_epoch(model, dataloader, 1)

        # then
        assert runner._optimizer.zero_grad.call_count == len(dataloader)

    def test_run_epoch_calls_optimizer_step(self, runner, model, dataloader):
        # when
        runner.run_epoch(model, dataloader, 1)

        # then
        assert runner._optimizer.step.call_count == len(dataloader)

    def test_run_epoch_calls_backward(self, runner, model, dataloader):
        # given
        loss_mock = mock.Mock()
        loss_mock.__float__ = mock.Mock(return_value=0.0)
        criterion = mock.Mock(return_value=loss_mock)
        runner._criterion = criterion

        # when
        runner.run_epoch(model, dataloader, 1)

        # then
        assert loss_mock.backward.call_count == len(dataloader)

    def test_run_epoch_returns_losses_mean(self, runner, model, dataloader, loss_value):
        # when
        loss = runner.run_epoch(model, dataloader, 1)

        # then
        assert loss == loss_value


class TestValidationEpochRunner:
    @pytest.fixture
    def loss_value(self):
        return 5.0

    @pytest.fixture(scope="class")
    def dataset(self):
        return TensorInputLabelDataset()

    @pytest.fixture(scope="class")
    def dataloader(self):
        return data.DataLoader(TensorInputLabelDataset())

    @pytest.fixture
    def runner(self, loss_value):
        loss_mock = mock.Mock()
        loss_mock.__float__ = mock.Mock(return_value=loss_value)
        criterion = mock.Mock(return_value=loss_mock)
        runner = runners.ValidationEpochRunner(criterion)
        return runner

    def test_run_epoch_turns_off_train_mode(self, runner, model, dataloader):
        # when
        runner.run_epoch(model, dataloader, 1)

        # then
        model.train.assert_called_once_with(False)

    def test_run_epoch_returns_losses_mean(self, runner, model, dataloader, loss_value):
        # when
        val_loss = runner.run_epoch(model, dataloader, 1)

        # then
        assert val_loss == loss_value
