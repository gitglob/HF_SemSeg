import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_sorted_file_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    files.sort()
    return files

class KittiSemSegDataset(Dataset):
    """
    Dataset for KITTI semantic segmentation.
    Expects directory structure:
      root/
        image_2/         # input RGB images (.png)
        semantic/        # ground-truth labels (class IDs, .png)
        instance/        # instance masks (optional)
        semantic_rgb/    # colored labels (optional)
    """
    def __init__(self, root_dir, sequence="2013_05_28_drive_0000_sync",
                 train=True,
                 transform=None):
        self.root_dir = root_dir

        # Path to the match file
        self.match_file = os.path.join(root_dir, "data_2d_semantics", "train", "0000.txt")

        # Read the match file and extract image and mask paths
        all_images, all_masks = self._read_match_file(self.match_file)

        assert len(all_images) == len(all_masks), (
            f"Images ({len(all_images)}) vs masks ({len(all_masks)}) mismatch"
        )

        # 95/5 split idx
        split_idx = int(0.95 * len(all_images))

        if train:
            self.images = all_images[:split_idx]
            self.masks = all_masks[:split_idx]
        else:
            self.images = all_images[split_idx:]
            self.masks = all_masks[split_idx:]

        # Transforms
        self.transform = transform

        # Map original class IDs to new class IDs
        self.classes = [0, 6, 7, 8, 9, 10,
                        11, 12, 13, 17, 19,
                        20, 21, 22, 23, 26, 24, 25, 27, 28,
                        30, 32, 33, 34, 35, 36, 37, 38, 39, 
                        40, 41, 42, 44]
        self.class_mapping = {}
        for i in range(len(self.classes)):
            self.class_mapping[self.classes[i]] = i

    def _read_match_file(self, match_file):
        """
        Reads the match file and extracts image and mask paths.

        Args:
            match_file (str): Path to the match file.

        Returns:
            tuple: Two lists containing full paths to images and masks.
        """
        all_images = []
        all_masks = []

        with open(match_file, "r") as f:
            for line in f:
                image_path, mask_path = line.strip().split()
                all_images.append(os.path.join(self.root_dir, image_path))
                all_masks.append(os.path.join(self.root_dir, mask_path))

        return all_images, all_masks
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)
        for cid in np.unique(mask):
            if cid not in self.class_mapping:
                print(f"Warning: Class {cid} not in mapping.")
        mask = np.vectorize(self.class_mapping.get)(mask).astype(np.uint8)

        augmented = self.transform(image=image, mask=mask)
        image, mask = augmented['image'], augmented['mask']

        return image, mask

def plot_sample(img, mask):
    """Create a plot with image and mask side by side."""
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot image
    axes[0].imshow(img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask, cmap='nipy_spectral')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

# Example usage
if __name__ == '__main__':    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset_root = '/home/panos/Documents/data/kitti/data_semantics/training'
    transform = A.Compose([
        A.Resize(height=375,
                 width=1242,
                 interpolation=cv2.INTER_LINEAR),
        ToTensorV2()
    ])
    dataset = KittiSemSegDataset(dataset_root, train=False, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    for imgs, masks in loader:
        print(imgs.shape, masks.shape)
        # Take the first image and mask from the batch
        img = imgs[0].permute(1, 2, 0).cpu().numpy()
        mask = masks[0].squeeze(0).cpu().numpy()
        print(img.dtype, mask.dtype)
        
        # Denormalize and plot
        # img = denormalize_image(img)
        fig = plot_sample(img, mask)
        plt.show()
        break
