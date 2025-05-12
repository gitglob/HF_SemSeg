import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
import wandb
import os
import sys

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoSeg import DinoSeg
from data.dataset import KittiSemSegDataset
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
IMAGE_SIZE = (375, 1242)
NUM_CLASSES = 35
VIS_ATTENTION = False
checkpoint_path = "checkpoints/model1.pth"

# Initialize wandb
wandb.init(project="HF_SemSeg", 
           name="train_with_wandb", 
           config={
               "batch_size": BATCH_SIZE,
               "learning_rate": LEARNING_RATE,
               "num_epochs": NUM_EPOCHS,
               "image_size": IMAGE_SIZE,
               "num_classes": NUM_CLASSES
               }
)

def main():
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, target_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, target_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = DinoSeg(num_labels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load the best model if it exists
    start_epoch, best_val_miou = load_checkpoint(model, optimizer, checkpoint_path)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='macro',
        ignore_index=None
    ).to(device)

    for epoch in range(start_epoch + 1, NUM_EPOCHS + 1):
        ####### TRAINING #######
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Train")
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
            val_bar = tqdm(val_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}]  Val")
            for batch_idx, (imgs, masks) in enumerate(val_bar, start=1):
                imgs, masks = imgs.to(device), masks.to(device).squeeze(1)

                output = model(imgs, original_size=IMAGE_SIZE, return_attention=VIS_ATTENTION)
                if VIS_ATTENTION:
                    logits, attentions = output
                else:
                    logits = output
                loss = criterion(logits, masks.long())
                running_val_loss += loss.item()

                # compute IoU on this batch
                preds = torch.argmax(logits, dim=1)  # [B, H, W]
                miou_metric.update(preds, masks)

                # Log plots for the first batch
                if batch_idx == 1:
                    cls_map = None
                    if VIS_ATTENTION:
                        # Visualize attention map (attentions: (num_attention_layers, batch_size, num_heads, seq_len, seq_len))
                        att = attentions[-1]            # Get the attention map from the last attention layer (batch_size, num_heads, seq_len, seq_len)
                        att = att.mean(dim=1)           # Average over all attention heads (batch_size, seq_len, seq_len)
                        cls_map = att[0, 0, 1:]         # Get the attention of the [CLS] token to all image patches for the first image (seq_len-1)
                        cls_map = cls_map.reshape(IMAGE_SIZE[0] // model.patch_size, IMAGE_SIZE[1] // model.patch_size)  # Reshape to the logit size (H/patch, W/patch)
                        cls_map = cls_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension (1, 1, H/patch, W/patch)
                        cls_map = F.interpolate(cls_map, size=IMAGE_SIZE, mode="bilinear", align_corners=False) # Reshape to the original image size (H, W)
                        cls_map = cls_map.squeeze(0).squeeze(0)  # Remove batch and channel dimensions (H, W)
                        cls_map = cls_map.detach().cpu().numpy()

                    plot_image_and_masks(
                        imgs[0].permute(1, 2, 0).cpu().numpy(),  # Original image
                        masks[0].cpu().numpy(),                  # Ground truth
                        preds[0].cpu().numpy(),                  # Predicted segmentation
                        cls_map,                                 # Attention map
                        epoch
                    )

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
            save_checkpoint(model, optimizer, epoch, best_val_miou, checkpoint_path)

if __name__ == "__main__":
    main()