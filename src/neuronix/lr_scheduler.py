import math
from .optimizers import Optimizer
# TODO Test the LR Schedulers
class _LRScheduler:
    """
    Base class for all learninag rate schedulers.

    This class provides the basic structure and shared functionality
    for all schedulers. It takes an optimizer and provides a `step`
    method to update the learning rate.
    """
    def __init__(self, optimizer):
        if not isinstance(optimizer, object) or not hasattr(optimizer, 'param_groups'):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.last_epoch = -1
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def step(self):
        """
        Update the learning rate for each parameter group in the optimizer.
        """
        self.last_epoch += 1
        lrs = self.get_lr()
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = lrs[i]

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.__dict__.update(state_dict)


class StepLR(_LRScheduler):
    """
    Decays the learning rate of each parameter group by a factor of `gamma`
    every `step_size` epochs.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay. Defaults to 0.1.
    """
    def __init__(self, optimizer, step_size, gamma=0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        # Calculate new learning rate for each group
        return [base_lr * self.gamma**(self.last_epoch // self.step_size)
                for base_lr in [group['lr'] for group in self.optimizer.param_groups]]


class CosineAnnealingLR(_LRScheduler):
    """
    Set the learning rate of each parameter group using a cosine annealing schedule.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float, optional): Minimum learning rate. Defaults to 0.
    """
    def __init__(self, optimizer, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        # Cosine annealing formula
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in [group['lr'] for group in self.optimizer.param_groups]]


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str, optional): One of 'min', 'max'. In 'min' mode, lr is reduced when
                               the metric stops decreasing. Defaults to 'min'.
        factor (float, optional): Factor by which the learning rate will be reduced. Defaults to 0.1.
        patience (int, optional): Number of epochs with no improvement after which
                                  learning rate will be reduced. Defaults to 10.
        threshold (float, optional): Threshold for measuring the new optimum. Defaults to 1e-4.
    """
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.num_bad_epochs = 0
        self.best = float('inf') if self.mode == 'min' else float('-inf')
        self._is_better = self._compare_metrics

    def _compare_metrics(self, current_metric, best_metric):
        if self.mode == 'min':
            return current_metric < best_metric - self.threshold
        else: # mode == 'max'
            return current_metric > best_metric + self.threshold

    def step(self, metric):
        """
        To be called after each epoch.
        
        Args:
            metric (float): The metric to monitor (e.g., validation loss or accuracy).
        """
        if self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            for group in self.optimizer.param_groups:
                new_lr = group['lr'] * self.factor
                if new_lr > 1e-6: # Prevent learning rate from dropping to zero
                    group['lr'] = new_lr
            self.num_bad_epochs = 0
            
            
