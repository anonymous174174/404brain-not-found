import math
from .optimizers import Optimizer
# TODO Test the LR Schedulers
class _LRScheduler:
    """
    Base class for all learning rate schedulers.
    """

    __slots__ = ('optimizer', 'last_epoch', 'base_lrs')

    def __init__(self, optimizer, last_epoch = -1):
        if not isinstance(optimizer, Optimizer) or not hasattr(optimizer, 'param_groups'):
            raise TypeError(f"{type(optimizer).__name__} is not a valid Optimizer")

        self.optimizer = optimizer
        self.last_epoch = last_epoch

        # Store the initial learning rates for all parameter groups
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        # This ensures optimizer LRs match the schedule at creation (avoids off-by-one bug)
        self.step()

    def get_lr(self):
        """
        Must return a list of new learning rates for each param_group.
        """
        raise NotImplementedError

    def step(self):
        """
        Advances the scheduler by one epoch and updates learning rates.
        """
        if self.last_epoch == -1:
            # First call at initialization, set epoch to 0
            self.last_epoch = 0
        else:
            self.last_epoch += 1

        new_lrs = self.get_lr()
        for lr, group in zip(new_lrs, self.optimizer.param_groups):
            group['lr'] = lr



class StepLR(_LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    """

    __slots__ = ('step_size', 'gamma')

    def __init__(self, optimizer, step_size , gamma = 0.1, last_epoch = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class CosineAnnealingLR(_LRScheduler):
    """
    Cosine annealing learning rate schedule.
    """

    __slots__ = ('T_max', 'eta_min')

    def __init__(self, optimizer, T_max , eta_min = 0.0, last_epoch = -1):
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """

    __slots__ = ('optimizer', 'mode', 'factor', 'patience', 'threshold', 'threshold_mode',
                 'min_lr', 'cooldown', 'cooldown_counter', 'best', 'num_bad_epochs', 'mode_worse')

    def __init__(self, optimizer, mode = 'min', factor = 0.1, patience = 10,
                 threshold: float = 1e-4, threshold_mode: str = 'rel', min_lr: float = 0.0,
                 cooldown: int = 0):
        if not isinstance(optimizer, Optimizer) or not hasattr(optimizer, 'param_groups'):
            raise TypeError(f"{type(optimizer).__name__} is not a valid Optimizer")

        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.min_lr = min_lr
        self.cooldown = cooldown

        self.cooldown_counter = 0
        self.num_bad_epochs = 0

        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = float('-inf')

        self.best = self.mode_worse

    def step(self, metric: float):
        if self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore bad epochs during cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self):
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if new_lr < old_lr - 1e-12:  # Avoid floating point noise
                group['lr'] = new_lr

    def _is_better(self, current, best):
        if self.mode == 'min':
            if self.threshold_mode == 'rel':
                return current < best * (1 - self.threshold)
            else:  # 'abs'
                return current < best - self.threshold
        else:  # mode == 'max'
            if self.threshold_mode == 'rel':
                return current > best * (1 + self.threshold)
            else:  # 'abs'
                return current > best + self.threshold

