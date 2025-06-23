import sys
import cv2
from pathlib import Path
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
cur_dir     = Path(__file__).parent
project_dir = cur_dir.parent
sys.path.append(str(project_dir))

from models.DinoFPNbn import DinoFPN
from models.tools import CombinedLoss
from data.kitti360.dataset import KittiSemSegDataset
from data.kitti360.labels_kitti360 import trainId2label, NUM_CLASSES
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_and_validate(train_loader, val_loader, 
                       model, criterion, metric, 
                       optimizer, cfg, 
                       scheduler = None, 
                       start_epoch=1, 
                       best_val_miou=0.0,
                       device="cuda"):
    print(f"Starting training from epoch {start_epoch} until {cfg.train.num_epochs}")
    print(f"Using batch size {cfg.train.batch_size} with {cfg.train.accum_steps} accumulation steps...")
    for epoch in range(start_epoch, cfg.train.num_epochs + 1):
        ####### TRAINING #######
        model.train()
        metric.reset()
        optimizer.zero_grad()
        running_train_loss = 0.0

        # Iterate over the training dataset
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}] Train")
        for batch_idx, (imgs, masks) in enumerate(train_bar):
            imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            input = model.process(imgs).to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # forward
            logits = model(input)

            # loss
            masks = masks.to(device)  # [B, H, W]
            loss = criterion(logits, masks.long())

            # scale the loss down so that gradients accumulate correctly
            (loss / cfg.train.accum_steps).backward()

            # compute IoU
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            metric.update(preds, masks)

            # Step every accum_steps or at the end of epoch
            is_accum_step = (batch_idx + 1) % cfg.train.accum_steps == 0
            is_last_batch = batch_idx == len(train_loader) - 1

            # every accum_steps, step & zero_grad
            if is_accum_step or is_last_batch:
                optimizer.step()
                optimizer.zero_grad()

            # accumulate losses
            running_train_loss += loss.item()

            train_bar.set_postfix(loss=running_train_loss / (batch_idx + 1))

        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_miou = metric.compute().item()

        ####### VALIDATION #######
        model.eval()
        running_val_loss = 0.0
        metric.reset()

        # Prepare lists for storing predictions and targets for confusion matrix
        if cfg.wandb.enabled:
            all_preds = []
            all_targets = []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch}/{cfg.train.num_epochs}]  Val")
            for batch_idx, (imgs, masks) in enumerate(val_bar):
                imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                input = model.process(imgs).to(device)

                # forward + loss
                logits = model(input)
                cls_map = None

                # Loss
                masks = masks.to(device)  # [B, H, W]
                loss = criterion(logits, masks.long())

                # accumulate losses
                running_val_loss += loss.item()

                # compute IoU on this batch
                preds = torch.argmax(logits, dim=1)  # [B, H, W]
                metric.update(preds, masks)

                # Store predictions and targets for confusion matrix
                if cfg.wandb.enabled:
                    preds_np   = preds.view(-1).cpu().numpy()
                    targets_np = masks.view(-1).cpu().numpy()
                    valid_idx  = targets_np != 255       # drop ignore_index
                    all_preds.extend(preds_np[valid_idx].tolist())
                    all_targets.extend(targets_np[valid_idx].tolist())

                # Log plots for the first batch
                if batch_idx == 0 and cfg.wandb.enabled:
                    plot_image_and_masks(
                        imgs[0].permute(1, 2, 0).cpu().numpy(),  # Original image
                        masks[0].cpu().numpy(),                  # Ground truth
                        preds[0].cpu().numpy(),                  # Predicted segmentation
                        cls_map,                                 # Attention map
                        epoch, cfg.dataset.num_classes
                    )
                    if cfg.visualization.attention:
                        del attentions
                        torch.cuda.empty_cache()

                val_bar.set_postfix(val_loss=running_val_loss / (batch_idx + 1))

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = metric.compute().item()

        # Update learning rate based on validation loss
        if scheduler is not None:
            scheduler.step()

        ####### LOG TO W&B #######
        if cfg.wandb.enabled:
            # Log confusion matrix
            class_names = [trainId2label[i].name for i in range(NUM_CLASSES)]
            confmat = wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_targets,
                preds=all_preds,
                class_names=class_names
            )

            # Log metrics
            wandb.log({
                "Train Loss": avg_train_loss,
                "Train mIoU": avg_train_miou,
                "Epoch": epoch,
                "Validation Loss": avg_val_loss,
                "Validation mIoU": avg_val_miou,
                "Learning Rate": optimizer.param_groups[0]['lr'],
                "Validation Confusion Matrix": confmat
            })

        ####### PRINT & CHECKPOINT #######
        print(
            f"Epoch {epoch:02d} | Learning Rate: {optimizer.param_groups[0]['lr']:.6f} | "
            f"\n  Train Loss: {avg_train_loss:.4f} | mIoU: {avg_train_miou:.4f} "
            f"\n  Val   Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}"
        )

        # Save best model
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            save_checkpoint(model, optimizer, epoch, best_val_miou, cfg.checkpoint, scheduler)

def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    train_transform = A.Compose([
        # -- Geometric --
        A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0), # preserve scale/context
        A.Affine(
            # translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # ±5% shift
            scale=(0.8, 1.0),                                           # zoom between 0.8×–1.0×
            rotate=(-3, 3),                                             # ±3° roll
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            fill_mask=255,
            p=0.7
        ),
        A.Perspective(scale=(0.01, 0.03), p=0.5),  # tiny camera viewpoint warp

        # -- Photometric --
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.05, 0.2), p=1.0),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=1.0),
            A.RandomSunFlare(src_radius=50, p=1.0)
        ], p=0.5),

        # -- Occlusions --
        A.CoarseDropout(num_holes_range=(1, 4), 
                        hole_height_range=(5, 30), 
                        hole_width_range=(5, 30), 
                        p=0.5),                                # random occlusion

        # — Blur & noise: motion, sensor, compression —
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),
            A.GaussianBlur(blur_limit=(3,5), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.5),
        A.GaussNoise(
            std_range=(10.0/255.0, 50.0/255.0),
            mean_range=(0.0, 0.0),
            p=0.5
        )
    ])

    # Define deterministic transforms for validation
    val_transform = A.Compose([
        A.CenterCrop(height=crop_size[0], width=crop_size[1])
    ])

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, 
                              shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model, loss function, and optimizer
    model = DinoFPN(
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
    start_epoch, best_val_miou = load_checkpoint(model, optimizer, cfg.checkpoint, scheduler)

    train_and_validate(
        train_loader, val_loader, 
        model, criterion, miou_metric,
        optimizer, scheduler, 
        cfg, 
        start_epoch=start_epoch, 
        best_val_miou=best_val_miou,
        device=device
    )

if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
