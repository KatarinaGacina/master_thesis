import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELossMasked:
    def __init__(self, pos_weight=None):
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        
    def compute_loss(self, preds, labels, mask):
        mask = mask[:, 0, 0, :]
        mask = mask.float()

        assert preds.shape == mask.shape
        assert preds.shape == labels.shape
        
        loss = self.criterion(preds, labels)
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)