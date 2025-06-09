import os
import sys
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
import albumentations as A
import wandb
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoFPNbn import DinoFPN as DinoSeg
from models.tools import CombinedLoss
from data.dataset import KittiSemSegDataset
from data.labels_kitti360 import trainId2label, NUM_CLASSES
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main(cfg: DictConfig):
    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    val_transform = A.Compose([
        A.CenterCrop(crop_size[0], crop_size[1])
    ])

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model, loss function, and optimizer
    model = DinoSeg(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )
    model = model.to(device)
    criterion = CombinedLoss(alpha=0.8, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='micro',
        ignore_index=255
    ).to(device)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)

    # Load the best model if it exists
    load_checkpoint(model, optimizer, cfg.checkpoint, scheduler)

    ####### VALIDATION #######
    model.eval()
    running_val_loss = 0.0
    miou_metric.reset()

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(val_loader)):
            imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            imgs = model.process(imgs)

            # forward + loss
            logits = model(imgs)

            # Loss
            masks = masks.to(device)  # [B, H, W]
            loss = criterion(logits, masks.long())

            # accumulate losses
            running_val_loss += loss.item()

            # compute IoU on this batch
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            miou_metric.update(preds, masks)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = miou_metric.compute().item()

        ####### PRINT & CHECKPOINT #######
        print(f"\n  Val   Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}")

if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
