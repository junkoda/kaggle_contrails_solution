import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class BCELoss(_Loss):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,
                y_sym_pred: torch.Tensor, y_sym: torch.Tensor,
                y_pred: torch.Tensor, y: torch.Tensor, w) -> torch.Tensor:
        loss_sym = self.criterion(y_sym_pred, y_sym).mean(dim=(1, 2, 3))
        loss_original = self.criterion(y_pred, y).mean(dim=(1, 2, 3))
        loss = loss_sym + w * loss_original

        return loss.mean()  # mean of batch


class DiceLoss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self,
                y_sym_pred: torch.Tensor, y_sym: torch.Tensor,
                y_pred: torch.Tensor, y: torch.Tensor, w,
                *, eps=1e-6) -> torch.Tensor:
        """
        Args:
          y_pred (Tensor): logit (batch_size, 1, H, W)
          y (Tensor): target in [0, 1]
          w (float or Tensor): (batch_size) w=1 for non-augmented subset
        """
        y_sym_pred = F.logsigmoid(y_sym_pred).exp()
        y_pred = F.logsigmoid(y_pred).exp()

        tp_sym = torch.sum(y_sym_pred * y_sym)
        tp = torch.sum(w * torch.sum(y_pred * y, dim=(1, 2, 3)))
        denom_sym = torch.sum(y_sym_pred + y_sym)
        denom = torch.sum(w * torch.sum(y_pred + y, dim=(1, 2, 3)))
        dice_score = 2 * (tp_sym + tp) / (denom_sym + denom).clamp_min(eps)

        return 1.0 - dice_score
