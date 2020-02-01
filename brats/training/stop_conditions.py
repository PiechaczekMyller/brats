class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.epochs_without_improvement = 0
        self.best_loss = None

    def update(self, loss):
        if self.best_loss is not None:
            if loss <= self.best_loss:
                self.best_loss = loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        else:
            self.best_loss = loss

    def check_stop_condition(self):
        return self.patience < self.epochs_without_improvement
