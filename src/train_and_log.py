import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
import wandb
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoSeg import DinoSeg
from data.dataset import KittiSemSegDataset
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint, get_cls_attention_map


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main(cfg: DictConfig):
    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    IMAGE_SIZE = (cfg.dataset.H, cfg.dataset.W)

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, target_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, 
                              shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, target_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, 
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = DinoSeg(
        num_labels=cfg.dataset.num_classes, 
        freeze_backbone=cfg.model.freeze_backbone
    )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Load the best model if it exists
    start_epoch, best_val_miou = load_checkpoint(model, optimizer, cfg.checkpoint)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='macro',
        ignore_index=None
    ).to(device)

    for epoch in range(start_epoch + 1, cfg.train.num_epochs + 1):
        ####### TRAINING #######
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{cfg.train.num_epochs}] Train")
        for batch_idx, (imgs, masks) in enumerate(train_bar, start=1):
            imgs, masks = imgs.to(device), masks.to(device).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # forward + loss
            outputs = model(imgs, original_size=IMAGE_SIZE)
            loss = criterion(outputs, masks.long())

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss / batch_idx)

        avg_train_loss = running_loss / len(train_loader)

        ####### VALIDATION #######
        model.eval()
        running_val_loss = 0.0
        miou_metric.reset()

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch}/{cfg.train.num_epochs}]  Val")
            for batch_idx, (imgs, masks) in enumerate(val_bar, start=1):
                imgs, masks = imgs.to(device), masks.to(device).squeeze(1)

                # forward + loss
                output = model(imgs, original_size=IMAGE_SIZE, 
                               return_attention=batch_idx==1)
                if batch_idx==1:
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
                if batch_idx == 1:
                    plot_image_and_masks(
                        imgs[0].permute(1, 2, 0).cpu().numpy(),  # Original image
                        masks[0].cpu().numpy(),                  # Ground truth
                        preds[0].cpu().numpy(),                  # Predicted segmentation
                        cls_map,                                 # Attention map
                        epoch
                    )
                    del attentions
                    torch.cuda.empty_cache()

                val_bar.set_postfix(val_loss=running_val_loss / batch_idx)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = miou_metric.compute().item()

        ####### LOG TO W&B #######
        wandb.log({
            "Train Loss": avg_train_loss,
            "Validation Loss": avg_val_loss,
            "Validation mIoU": avg_val_miou,
            "Epoch": epoch
        })

        ####### LOG & CHECKPOINT #######
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val   Loss: {avg_val_loss:.4f} | "
            f"Val  mIoU: {avg_val_miou:.4f}"
        )

        # Save best model
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            save_checkpoint(model, optimizer, epoch, best_val_miou, cfg.checkpoint)

if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
