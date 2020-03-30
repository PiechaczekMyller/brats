import pytest
from brats.training import fake_apex


class TestScaleLoss:
    @pytest.mark.parametrize("args, kwargs",
                             [(list(), dict()),
                              ([1, "test", 1.2], dict()),
                              (list(), {"arg1": 1, "arg2": "test", "arg3": 1.2}),
                              ([1, "test", 1.2], {"arg1": 1, "arg2": "test", "arg3": 1.2})])
    def test_if_raises(self, args, kwargs):
        with pytest.raises(RuntimeError):
            fake_apex.scale_loss(*args, **kwargs)


class TestInitialize:
    @pytest.mark.parametrize("args, kwargs",
                             [(list(), dict()),
                              ([1, "test", 1.2], dict()),
                              (list(), {"arg1": 1, "arg2": "test", "arg3": 1.2}),
                              ([1, "test", 1.2], {"arg1": 1, "arg2": "test", "arg3": 1.2})])
    def test_if_raises(self, args, kwargs):
        with pytest.raises(RuntimeError):
            fake_apex.scale_loss(*args, **kwargs)
