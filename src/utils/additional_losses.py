import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLossMasked(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()

        self.smooth = smooth

    def forward(self, logits, labels, mask=None):

        logits = torch.sigmoid(logits)

        logits = logits.float()
        labels = labels.float()

        if mask is not None:
            assert mask.shape == logits.shape

            logits = logits * mask
            labels = labels * mask

        intersection = (logits * labels).sum(dim=1)
        denominator = logits.sum(dim=1) + labels.sum(dim=1)

        dice_score = (2 * intersection + self.smooth) / (denominator + self.smooth)

        return 1 - dice_score.mean()


class FocalLossMasked(nn.Module):
    def __init__(self, alpha=0.8, gamma=1.0, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.reduction = reduction

    def forward(self, logits, labels, mask):
        probs = torch.sigmoid(logits)

        p_t = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)

        if mask is not None:
            assert mask.shape == logits.shape
            
            loss = loss * mask

        if self.reduction == 'mean':
            if mask is not None:
                return loss.sum() / (mask.sum() + 1e-8)
            else:
                return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss