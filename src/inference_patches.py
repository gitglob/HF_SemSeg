from types import SimpleNamespace
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import os
import sys
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoFPN import DinoFPN
from data.kitti360.dataset import KittiSemSegDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 1
PATCH_SIZE = (224, 224)  # Same as the patch size used during training
IMAGE_SIZE = (375, 1242)
NUM_CLASSES = 33
checkpoint_path = "checkpoints/dino-unet-kitti360-deep.pth"

# Define deterministic transforms for validation
transform = A.Compose([ToTensorV2()])

def infer_on_patches(model, image, patch_size, num_classes):
    """
    Perform sliding-window inference covering all pixels, with overlap if needed.

    Args:
        model: The segmentation model.
        image: Input tensor [1, C, H, W].
        patch_size: Tuple (patch_height, patch_width).
        num_classes: Number of segmentation classes.

    Returns:
        full_pred: [H, W] predicted class mask.
        patches: list of extracted patches for visualization.
    """
    _, _, H, W = image.shape
    patch_H, patch_W = patch_size

    # compute start positions to ensure full coverage
    def get_starts(dim, patch):
        starts = list(range(0, dim, patch))
        if starts[-1] + patch < dim:
            starts.append(dim - patch)
        return starts

    row_starts = get_starts(H, patch_H)
    col_starts = get_starts(W, patch_W)

    # containers for logits and counts
    logits_full = torch.zeros((1, num_classes, H, W), device=image.device)
    count_mask = torch.zeros((1, 1, H, W), device=image.device)

    patches = []
    model.eval()
    with torch.no_grad():
        for i in row_starts:
            i_start = i
            for j in col_starts:
                j_start = j

                # calculate patch boundaries
                i_end = i + patch_H if i + patch_H < H else H
                j_end = j + patch_W if j + patch_W < W else W

                # Check if patch exceeds image dimensions
                if i_end > H:
                    i_start = i_end - patch_H
                if j_end > W:
                    j_start = j_end - patch_W

                # extract patch, pad if at edge
                patch = image[:, :, i_start:i_end, j_start:j_end]
                pad_h = max(0, patch_H - (i_end - i_start))
                pad_w = max(0, patch_W - (j_end - j_start))
                if pad_h > 0 or pad_w > 0:
                    patch = F.pad(patch, (0, pad_w, 0, pad_h))

                patches.append(patch.squeeze(0).permute(1, 2, 0).cpu().numpy())

                # forward and crop logits to original patch area
                logits_patch = model(patch)
                logits_patch = logits_patch[:, :, :patch_H - pad_h, :patch_W - pad_w]

                # accumulate logits and counts
                logits_full[:, :, i_start:i_end, j_start:j_end] += logits_patch
                count_mask[:, :, i_start:i_end, j_start:j_end] += 1

    # average overlapping logits
    avg_logits = logits_full / count_mask
    full_pred = torch.argmax(avg_logits, dim=1).squeeze(0)
    return logits_full, full_pred, patches


def main():
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model
    model_cfg = SimpleNamespace(
        backbone="facebook/dinov2-base",
        freeze_backbone=True
    )
    model = DinoFPN(num_labels=NUM_CLASSES, model_cfg=model_cfg).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Metric: mean IoU
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='micro',
        ignore_index=None
    ).to(device)

    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # Perform inference on patches and merge results
            logits, preds, patches = infer_on_patches(model, imgs, PATCH_SIZE, NUM_CLASSES)
            assert(masks.max() < logits.shape[1])

            # Compute mIoU for the current image
            miou_metric.reset()
            miou_metric.update(preds.unsqueeze(0), masks)
            miou = miou_metric.compute().item()

            # Plot the patches
            fig, axes = plt.subplots(1, len(patches), figsize=(15, 5))
            for ax, patch in zip(axes, patches):
                ax.imshow(patch)
                ax.axis("off")
            plt.suptitle("Image Patches")
            plt.show()

            # Plot the image, ground truth, and prediction
            img_np = imgs[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            mask_np = masks[0].cpu().numpy()
            pred_np = preds.cpu().numpy()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(img_np)
            ax1.set_title("Input Image")
            ax1.axis("off")

            ax2.imshow(mask_np, cmap="viridis")
            ax2.set_title("Ground Truth")
            ax2.axis("off")

            ax3.imshow(pred_np, cmap="viridis")
            ax3.set_title(f"Prediction (mIoU={miou:.4f})")
            ax3.axis("off")

            plt.tight_layout()
            plt.show()  # Wait for the user to close the plot before continuing


if __name__ == "__main__":
    main()
