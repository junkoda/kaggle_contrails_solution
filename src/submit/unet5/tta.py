"""
Test-time augmentation (TTA)
"""
import torch

_tta_config = {'d4prob': (8, True), 'd4logit': (8, False),
               'rotprob': (4, True), 'rotlogit': (4, False), 'none': (1, False)}


def _tta_stack(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Increate input x by n TTA patterns
    batch_size -> n * batch_size
    """
    stack = []
    for k in range(4):
        xa = torch.rot90(x, k, dims=[2, 3])
        stack.append(xa)
        if n == 8:
            stack.append(torch.flip(xa, dims=[3, ]))
    return torch.cat(stack, dim=0)


def _tta_average(y_pred: torch.Tensor, n: int, prob: bool) -> torch.Tensor:
    """
    Average TTA augmented predictions
    n * batch_size -> batch_size
    """
    batch_size, nch, H, W = y_pred.shape
    y_pred = y_pred.view(n, batch_size // n, nch, H, W)
    batch_size = batch_size // n
    y_avg = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=y_pred.device)

    if prob:
        y_pred = y_pred.sigmoid()

    if n == 4:
        for k in range(4):
            y_avg += (1 / n) * torch.rot90(y_pred[k], -k, dims=[2, 3])
    if n == 8:
        for k in range(4):
            y_avg += (1 / n) * torch.rot90(y_pred[2 * k], -k, dims=[2, 3])
            y_avg += (1 / n) * torch.rot90(y_pred[2 * k + 1].flip(dims=[3, ]), -k, dims=(2, 3))

    if prob:
        y_avg = y_avg.clamp(1e-6, 1 - 1e-6)
        return y_avg.logit()
    return y_avg


class TTA:
    """
    Args:
      tta_str: d4prob, d4logit, rotprob, rotlogit
    """
    def __init__(self, tta_str: str):        
        self.n, self.prob = _tta_config[tta_str]
    
    def __repr__(self):
        return 'TTA(n={}, prob={})'.format(self.n, self.prob)

    def stack(self, x):
        if self.n == 1:
            return x
        else:
            return _tta_stack(x, self.n)

    def average(self, y):
        if self.n == 1:
            return y
        else:
            return _tta_average(y, self.n, self.prob)
