import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import os
import sys

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoSeg import DinoSeg
from data.dataset import KittiSemSegDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 1
IMAGE_SIZE = (375, 1242)
NUM_CLASSES = 35
checkpoint_path = "checkpoints/best_model.pth"

def main():
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    val_dataset = KittiSemSegDataset(dataset_root, train=True, target_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model
    model = DinoSeg(num_labels=NUM_CLASSES).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Metric: mean IoU
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='macro',
        ignore_index=None
    ).to(device)

    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(val_loader):
            imgs, masks = imgs.to(device), masks.to(device).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # Forward pass
            outputs = model(imgs, original_size=IMAGE_SIZE)
            preds = torch.argmax(outputs, dim=1)  # [B, H, W]

            # Compute mIoU for the current image
            miou_metric.reset()
            miou_metric.update(preds, masks)
            miou = miou_metric.compute().item()

            # Plot the image, ground truth, and prediction
            img_np = imgs[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            mask_np = masks[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

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