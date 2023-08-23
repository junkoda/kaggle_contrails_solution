"""
Custom scheduler

configuration
  - linear:
      epoch_end: 3
      lr_start: 0
      lr_end: 2e-3
  - const:
      epoch_end: 4
  - cosine:
      epoch_end: 12
      lr_end: 1e-5
  - const:
      epoch_end: 14
"""
import math
from torch.optim.optimizer import Optimizer


class _Const:
    """
    - const:
        epoch_end: 2
        (optional) lr: 2e-3
    """
    def __init__(self, lr: float, epoch: float, params: dict):
        """
        Args
          lr (float): current learning rate
          epoch (float): current epoch
          params (dict): schedule parameters
        """
        self.start = epoch
        self.end = params['epoch_end']
        self.lr = float(params.get('lr', lr))
        assert self.start < self.end

    def __contains__(self, epoch) -> bool:
        # Return if the epoch is in the range this schedule is supposed to work
        return self.start <= epoch < self.end

    def __call__(self, epoch: float) -> float:
        # Return learning rate at given epoch
        return self.lr

    def next(self):
        # Return epoch and lr at the end of this schedule,
        # passing them to the next scheduler
        return self.end, self.lr


class _Linear:
    """
    - linear:
        epoch_end: 4
        lr_end: 2e-3
        (optional) lr_start: 5e-5
    """
    def __init__(self, lr: float, epoch: float, params: dict):
        self.start = epoch
        self.end = params['epoch_end']
        self.lr_start = float(params.get('lr_start', lr))
        self.lr_end = float(params['lr_end'])
        assert self.start < self.end

    def __contains__(self, epoch: float) -> float:
        return self.start <= epoch < self.end

    def __call__(self, epoch: float) -> float:
        x = (epoch - self.start) / (self.end - self.start)
        return self.lr_start + x * (self.lr_end - self.lr_start)

    def next(self):
        return self.end, self.lr_end


class _Cosine:
    """
    - cosine:
        epoch_end: 12
        lr_min: 1e-6
        (optional): lr_start 2e-3
        (optional): cycle: 180 (360 for full period of cosine)
    """
    def __init__(self, lr: float, epoch: float, params: dict):
        self.start = epoch
        self.end = params['epoch_end']
        self.lr_start = float(params.get('lr_start', lr))
        self.lr_end = float(params.get('lr_end', 0))
        self.fac = math.pi * (float(params.get('cycle', 180)) / 180)
        assert self.start < self.end

    def __contains__(self, epoch) -> bool:
        return self.start <= epoch < self.end

    def __call__(self, epoch: float) -> float:
        x = (epoch - self.start) / (self.end - self.start)
        amp = 0.5 * (self.lr_start - self.lr_end)
        return self.lr_end + amp * (1 + math.cos(self.fac * x))

    def next(self) -> tuple:
        return self.end, self.lr_end


class _ReduceOnPlateau:
    """
    Reduce learning rate by a given factor when loss is not improving

    - reduce_on_plateau:
        lr_start: 1e-2
        epoch_end: 4
        patience: 2      # do not reduce lr for 2 times if loss does not improve
        factor: 0.5      # multiply lr by factor
    """
    def __init__(self, lr: float, epoch: float, params: dict):
        self.start = epoch
        self.end = params['epoch_end']
        self.lr_start = float(params.get('lr_start', lr))
        self.lr_end = self.lr_start

        # ReduceOnPlateau parameters
        self.patience = params['patience']
        self.factor = params['factor']
        self.lr = self.lr_start

        # Internal variables
        self.best_loss = 1e8
        self.count_no_improvement = 0
        assert self.start < self.end

    def __contains__(self, epoch: float) -> bool:
        return self.start <= epoch < self.end

    def __call__(self, epoch: float) -> float:
        return self.lr

    def next(self) -> tuple:
        return self.end, self.lr_end

    def update_loss(self, epoch: float, loss: float) -> None:
        """
        Update loss and update learning rate depending on the loss
        """
        eps = 1e-4
        if loss < (1 - eps) * self.best_loss:
            self.best_loss = loss
            self.count_no_improvement = 0
        else:
            self.count_no_improvement += 1
            if self.count_no_improvement > self.patience:
                self.lr *= self.factor
                self.count_no_improvement = 0


class Scheduler(object):
    def __init__(self, optimizer: Optimizer, cfg: dict):
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.base_lr = optimizer.param_groups[0]['lr']

        # Initialize base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]

        self._init_schedules(cfg)
        self.step(0)

    def __len__(self) -> int:
        return math.ceil(self.epoch_end)

    def _init_schedules(self, learning_rate: list[dict]):
        """
        Arg:
          learning_rate (list[dict]): list of schedules
        """
        self.schedules = []
        epoch = 0

        lr = self.base_lr

        for d in learning_rate:
            assert len(d) == 1
            for name, params in d.items():
                if name == 'const':
                    schedule = _Const(lr, epoch, params)
                elif name == 'cosine':
                    schedule = _Cosine(lr, epoch, params)
                elif name == 'linear':
                    schedule = _Linear(lr, epoch, params)
                elif name == 'reduce_on_plateau':
                    schedule = _ReduceOnPlateau(lr, epoch, params)
                else:
                    raise ValueError('Unknown schedule: {}'.format(name))

                epoch, lr = schedule.next()
                self.schedules.append(schedule)

        self.epoch_end = epoch

    def step(self, epoch: float):
        lr_factor = self._get_lr(epoch) / self.base_lr

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * lr_factor

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _get_lr(self, epoch: float) -> float:
        for schedule in self.schedules:
            if epoch in schedule:
                return schedule(epoch)

        end, lr = self.schedules[-1].next()

        if epoch > 1.001 * end:
            raise ValueError('Epoch {} is out of range'.format(epoch))

        return lr

    def get_last_lr(self) -> float:
        return self._last_lr

    def update_loss(self, epoch: float, loss: float) -> None:
        for schedule in self.schedules:
            if epoch in schedule and isinstance(schedule, _ReduceOnPlateau):
                return schedule.update_loss(epoch, loss)
