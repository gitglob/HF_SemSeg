import torch


def compute_iou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    preds: (B, C, H, W) raw logits or softmax probabilities
    labels: (B, H, W) integer class labels in [0..num_classes-1]
    """
    # 1) get per-pixel class predictions
    preds = torch.argmax(preds, dim=1)  # (B, H, W)

    ious = []
    for cls in range(num_classes):
        # mask pixels of this class
        pred_inds  = (preds == cls)
        label_inds = (labels == cls)
        # intersection & union
        intersection = (pred_inds & label_inds).sum().float()
        union        = (pred_inds | label_inds).sum().float()
        if union == 0:
            ious.append(float('nan'))   # no pixels of this class in GT + pred
        else:
            ious.append((intersection / union).item())

    # mean over classes, ignoring NaNs
    return float(torch.tensor(ious).nanmean())
