import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        """
        logits: [B, C, H, W], raw scores
        target: [B, H, W], int64 class labels
        """
        num_classes = logits.shape[1]
        # softmax → [B, C, H, W]
        probs = F.softmax(logits, dim=1)
        # one‐hot → [B, C, H, W]
        target_onehot = F.one_hot(target, num_classes).permute(0,3,1,2).float()
        
        # intersection & union per class
        inter = (probs * target_onehot).sum(dim=(2,3))
        union = probs.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
        
        dice_score = (2 * inter + self.eps) / (union + self.eps)
        # we want a minimizable loss
        return 1 - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return loss_ce + self.dice_weight * loss_dice