class EarlyStopping:
    """
    Early stopping stop condition.
    Args:
        :param patience: Allowed number of epochs without an improvement.
    """

    def __init__(self, patience):
        self.patience = patience
        self.epochs_without_improvement = 0
        self.best_loss = None

    def _update(self, loss):
        """
        Method to update the state of the stop condition before checking it.
        Args:
            :param loss: Current loss of the model if it is higher than best, 1 is added to the counter.
        """
        if self.best_loss is not None:
            if loss <= self.best_loss:
                self.best_loss = loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        else:
            self.best_loss = loss

    def check_stop_condition(self, loss):
        """
        Function to check whether the stop condition is met.
        Args:
            :param loss: Current loss of the model.
        """
        self._update(loss)
        return self.patience < self.epochs_without_improvement
