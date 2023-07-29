import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model import F_LOSS_ALPHA, F_LOSS_GAMMA


class FocalLoss(nn.Module):
    def __init__(self, loss_func, gamma: float=1.5, alpha: float=0.25) -> None:
        super().__init__()
        self._loss_func = loss_func
        self._gamma = gamma
        self._alpha = alpha
        self._reduction = loss_func.reduction
        self._loss_func.reduction = "none"
    
    def forward(self, pred, gtrue):
        loss = self._loss_func(pred, gtrue)
        pred_prob = torch.sigmoid(pred)
        p_t = gtrue * pred_prob + (1 - gtrue) * (1 - pred_prob)
        loss = loss * ((1.0 - p_t) ** self._gamma)
        if self._alpha > 0:
            alpha_t = self._alpha * gtrue + (1 - self._alpha) * (1 - gtrue)
            loss = alpha_t * loss
        if self._reduction == "mean":
            loss = loss.mean()
        elif self._reduction == "sum":
            loss = loss.sum()
        return loss

class Part1Loss(nn.Module): # [ROTATOR CUFF, RIGHT/LEFT, SAG/COR, 0, 0, 0]
    def __init__(self, enable_focal_loss: bool) -> None:
        super().__init__()
        self._rc_loss_func = nn.BCEWithLogitsLoss()
        self._rl_loss_func = nn.BCEWithLogitsLoss()
        self._sc_loss_func = nn.BCEWithLogitsLoss()
        self._tc1_loss_func = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, gtrue):
        rc_loss = self._rc_loss_func(pred[:, 0], gtrue[:, 0])
        if abs(rc_loss - 1e-5) < 1e-7:
            rc_loss.data.zero_()
            if rc_loss.grad is not None:
                rc_loss.grad.zero_()
        rl_loss = self._rl_loss_func(pred[:, 1], gtrue[:, 1])
        if abs(rl_loss - 1e-5) < 1e-7:
            rl_loss.data.zero_()
            if rl_loss.grad is not None:
                rl_loss.grad.zero_()
        sc_loss = self._sc_loss_func(pred[:, 2], gtrue[:, 2])
        if abs(sc_loss - 1e-5) < 1e-7:
            sc_loss.data.zero_()
            if sc_loss.grad is not None:
                sc_loss.grad.zero_()
        tc1_loss = self._tc1_loss_func(pred[:, 3:], gtrue[:, 3:])
        return 1/2 * rc_loss + 1/2 *  rl_loss + 1/2 *  sc_loss + tc1_loss


class Part2Loss(nn.Module):
    def __init__(self, label_smooth=None, class_num=137, weight=None):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num
        self.weight = weight

    def forward(self, pred, target):
        eps = 1e-12
        if self.label_smooth is not None:
            logprobs = F.log_softmax(pred, dim=1)
            # target = F.one_hot(target, self.class_num)	# 转换成one-hot
            target = torch.clamp(target.float(), min=self.label_smooth/(self.class_num-1), max=1.0-self.label_smooth)
            if self.weight is not None:
                loss = -1*torch.sum(target*logprobs*self.weight, 1)
            else:
                loss = -1*torch.sum(target*logprobs, 1)
        else:
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()