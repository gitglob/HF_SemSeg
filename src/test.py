import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import os
import sys
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoFPNbn import DinoFPN
from data.dataset import KittiSemSegDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 1
NUM_CLASSES = 33

def main(cfg: DictConfig):
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    cmap = plt.get_cmap("viridis", NUM_CLASSES)
    norm = BoundaryNorm(boundaries=np.arange(NUM_CLASSES + 1) - 0.5, ncolors=NUM_CLASSES)

    # Initialize model
    model = DinoFPN(num_labels=cfg.dataset.num_classes, model_cfg=cfg.model).to(device)
    checkpoint_path = f"checkpoints/{cfg.checkpoint.model_name}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
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
            masks = masks.to(device)  # [B, H, W]
            imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            print(f"\nProcessing image {idx + 1}/{len(val_loader)}")

            # Forward pass with inference time
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(imgs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            inference_time = t1 - t0
            print(f"Inference time: {inference_time:.4f} sec") # ~0.45 seconds

            preds = torch.argmax(logits, dim=1)  # [B, H, W]

            # Compute mIoU for the current image
            miou_metric.reset()
            miou_metric.update(preds, masks)
            miou = miou_metric.compute().item()

            print(f"mIoU = {miou:.4f}")
            num_identical = torch.sum(preds == masks).item()  
            print(f"Number of identical pixels: {num_identical}")
            num_non_identical = torch.sum(preds != masks).item()
            print(f"Number of non-identical pixels: {num_non_identical}")
            print(f"Accuracy: {num_identical / (num_identical + num_non_identical):.4f}")

            # Plot the image, ground truth, and prediction
            img_np = imgs[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
            mask_np = masks[0].cpu().numpy()
            pred_np = preds[0].cpu().numpy()

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(img_np)
            ax1.set_title("Input Image")
            ax1.axis("off")

            ax2.imshow(mask_np, cmap=cmap, norm=norm)
            ax2.set_title("Ground Truth")
            ax2.axis("off")

            ax3.imshow(pred_np, cmap=cmap, norm=norm)
            ax3.set_title(f"Prediction (mIoU={miou:.4f})")
            ax3.axis("off")

            plt.tight_layout()
            plt.show()  # Wait for the user to close the plot before continuing


if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="test"
    ):
        cfg = compose(config_name="config")
        main(cfg)