from brats.training import stop


class TestValidationLossDidNotImprove:

    def test_is_satisfied_when_not_improved(self):
        # given
        condition = stop.ValidationLossDidNotImprove(patience=5, min_delta=1.0)
        val_losses = [1.5, 1.3, 1.4, 0.8, 0.6]

        # when
        for epoch, val_loss in enumerate(val_losses[:-1]):
            assert condition.is_condition_satisfied(epoch, 0.0, val_loss) is False

        # then
        assert condition.is_condition_satisfied(0, 0.0, val_losses[-1]) is True

    def test_is_not_satisfied_when_improved(self):
        # given
        condition = stop.ValidationLossDidNotImprove(patience=5, min_delta=0.8)
        val_losses = [1.5, 1.3, 1.4, 0.8, 0.6]

        # when
        for epoch, val_loss in enumerate(val_losses[:-1]):
            assert condition.is_condition_satisfied(epoch, 0.0, val_loss) is False

        # then
        assert condition.is_condition_satisfied(0, 0.0, val_losses[-1]) is False
