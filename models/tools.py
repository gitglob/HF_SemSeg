import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6, ignore_index: int = None):
        """
        Dice loss for multi‐class segmentation, with optional ignore‐index.
        Args:
            eps: small constant to avoid division by zero
            ignore_index: label value to ignore in both prediction and target
        """
        super().__init__()
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W] — raw, un‐normalised scores
        target: [B, H, W] — integer class labels in {0, …, C−1}, or ignore_index
        """
        B, C, H, W = logits.shape

        # 1) Build a mask of valid pixels
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)  # [B, H, W], bool
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        # 2) Remap ignored pixels to a valid class (e.g. 0) so one_hot won't crash
        target_safe = target.clone()
        target_safe[~valid_mask] = 0                  # doesn't matter which class

        # 3) One-hot encode
        probs = F.softmax(logits, dim=1)              # [B, C, H, W]
        target_onehot = F.one_hot(target_safe, C)     # [B, H, W, C]
        target_onehot = target_onehot.permute(0,3,1,2).float()  # [B, C, H, W]

        # 4) Zero out ignored pixels in both pred & gt
        mask = valid_mask.unsqueeze(1).float()        # [B, 1, H, W]
        probs = probs * mask
        target_onehot = target_onehot * mask

        # 5) Compute Dice
        inter = (probs * target_onehot).sum(dim=(2,3))               # [B, C]
        union = probs.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))  # [B, C]
        dice = (2*inter + self.eps) / (union + self.eps)             # [B, C]

        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor = None,
        alpha: float = 0.8,
        ignore_index: int = 255
    ):
        """
        Combined Cross‐Entropy + Dice loss.
        Args:
            weight: class‐weights for cross‐entropy
            alpha: weight for CE term (Dice is weighted by 1−alpha)
            ignore_index: label in target to ignore in both losses
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(eps=1e-6, ignore_index=ignore_index)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_dice
