import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm

import os
import sys

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoSeg import DinoSeg
from data.dataset import KittiSemSegDataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
IMAGE_SIZE = (375, 1242)

checkpoint_path = "checkpoints/best_model.pth"

def main():
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, target_size=IMAGE_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4, 
                            pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, target_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4, pin_memory=True)



    # Initialize model, loss function, and optimizer
    NUM_CLASSES = 35
    model = DinoSeg(num_labels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Load the best model if it exists
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model from {checkpoint_path}")

    # Metric: mean IoU over all classes (ignore_index for void if needed)
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES, 
        average='micro',       # mean over classes
        ignore_index=None      # or your void label
    ).to(device)

    best_val_miou = 0.0

    for epoch in range(1, NUM_EPOCHS+1):
        ####### TRAINING #######
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Train")
        for batch_idx, (imgs, masks) in enumerate(train_bar, start=1):
            imgs, masks = imgs.to(device), masks.to(device).squeeze(1)  # [B, 1, H, W] -> [B, H, W]

            # forward + loss
            logits = model(imgs)
            logits = F.interpolate(logits, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
            loss    = criterion(logits, masks.long())

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

                logits = model(imgs)
                logits = F.interpolate(logits, size=IMAGE_SIZE, mode="bilinear", align_corners=False)
                loss    = criterion(logits, masks.long())
                running_val_loss += loss.item()

                # compute IoU on this batch
                preds = torch.argmax(logits, dim=1)  # [B, H, W]
                miou_metric.update(preds, masks)

                val_bar.set_postfix(val_loss=running_val_loss / batch_idx)

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = miou_metric.compute().item()


        ####### LOG & CHECKPOINT #######
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val   Loss: {avg_val_loss:.4f} | "
            f"Val  mIoU: {avg_val_miou:.4f}"
        )

        # save best
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), checkpoint_path)
            print(f" → New best! Model saved (mIoU={best_val_miou:.4f})")

if __name__ == "__main__":
    main()