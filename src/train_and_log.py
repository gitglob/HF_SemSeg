import os
import sys
import numpy as np
import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoSeg import DinoSeg
from models.tools import CombinedLoss
from data.dataset import KittiSemSegDataset
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint, get_cls_attention_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    IMAGE_SIZE = (cfg.dataset.H, cfg.dataset.W)

    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    train_transform = A.Compose([
        # -- Geometric --
        A.RandomCrop(height=crop_size[0], width=crop_size[1]),    # preserve scale/context
        A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR, p=0.3), # slight tilt
        # A.Perspective(scale=(0.02, 0.05), p=0.3),                 # minor warp
        # A.Affine(scale=(0.8, 1.0), translate_percent=(0.1, 0.1), shear=10, p=0.5),

        # -- Photometric --
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(
            std_range=(10.0/255.0, 50.0/255.0),
            mean_range=(0.0, 0.0),
            p=0.3
        ), # sim camera noise
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.2),         # weather

        # -- Occlusions --
        A.CoarseDropout(num_holes_range=(1, 4), 
                        hole_height_range=(5, 30), 
                        hole_width_range=(5, 30), 
                        p=0.3),                                # random occlusion

        # -- Blur (optional) --
        # A.GaussianBlur(blur_limit=(3,7), p=0.2),

        # finally convert to tensor
        ToTensorV2()
    ])

    # Define deterministic transforms for validation
    val_transform = A.Compose([
        A.CenterCrop(height=crop_size[0], width=crop_size[1]),
        ToTensorV2()
    ])

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, 
                              shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = DinoSeg(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )
    model = model.to(device)
    criterion = CombinedLoss(dice_weight=1.0) # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='macro',
        ignore_index=None
    ).to(device)

    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Load the best model if it exists
    start_epoch, best_val_miou = load_checkpoint(model, optimizer, cfg.checkpoint, scheduler)

    for epoch in range(start_epoch + 1, cfg.train.num_epochs + 1):
        ####### TRAINING #######
        model.train()
        running_loss = 0.0
        miou_metric.reset()

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{cfg.train.num_epochs}] Train")
        for batch_idx, (imgs, masks) in enumerate(train_bar, start=1):
            imgs, masks = imgs.to(device), masks.to(device).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # forward + loss
            logits = model(imgs)
            loss = criterion(logits, masks.long())

            # compute IoU on this batch
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            miou_metric.update(preds, masks)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / batch_idx)

        avg_train_loss = running_loss / len(train_loader)
        avg_train_miou = miou_metric.compute().item()

        ####### VALIDATION #######
        model.eval()
        running_val_loss = 0.0
        miou_metric.reset()

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch}/{cfg.train.num_epochs}]  Val")
            for batch_idx, (imgs, masks) in enumerate(val_bar, start=1):
                imgs, masks = imgs.to(device), masks.to(device).squeeze(1)

                # forward + loss
                return_attention = batch_idx == 1 and cfg.visualization.attention
                output = model(imgs, return_attention=return_attention)
                if return_attention:
                    logits, attentions = output
                    cls_map = get_cls_attention_map(attentions, cfg.dataset.H, cfg.dataset.W, model.patch_size)
                else:
                    logits = output
                    cls_map = None
                loss = criterion(logits, masks.long())
                running_val_loss += loss.item()

                # compute IoU on this batch
                preds = torch.argmax(logits, dim=1)  # [B, H, W]
                miou_metric.update(preds, masks)

                # Log plots for the first batch
                if batch_idx == 1 and cfg.wandb.enabled:
                    plot_image_and_masks(
                        imgs[0].permute(1, 2, 0).cpu().numpy(),  # Original image
                        masks[0].cpu().numpy(),                  # Ground truth
                        preds[0].cpu().numpy(),                  # Predicted segmentation
                        cls_map,                                 # Attention map
                        epoch
                    )
                    if cfg.visualization.attention:
                        del attentions
                        torch.cuda.empty_cache()

                val_bar.set_postfix(val_loss=running_val_loss / batch_idx)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = miou_metric.compute().item()

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        ####### LOG TO W&B #######
        if cfg.wandb.enabled:
            wandb.log({
                "Train Loss": avg_train_loss,
                "Train mIoU": avg_train_miou,
                "Validation Loss": avg_val_loss,
                "Validation mIoU": avg_val_miou,
                "Epoch": epoch,
                "Learning Rate": optimizer.param_groups[0]['lr']
            })

        ####### LOG & CHECKPOINT #######
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | mIoU: {avg_train_miou:.4f} "
            f"Val   Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}"
        )

        # Save best model
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            save_checkpoint(model, optimizer, epoch, best_val_miou, cfg.checkpoint, scheduler)

if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
