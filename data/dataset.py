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

from .labels_kitti360 import id2label

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
                 transform=None,
                 debug=False):
        self.root_dir = root_dir

        # Path to the match file
        self.match_file = os.path.join(root_dir, "data_2d_semantics", "train", "0000.txt")

        # Read the match file and extract image and mask paths
        all_images, all_masks = self._read_match_file(self.match_file)

        # If in debug mode, only keep a random 10% of the dataset
        if debug:
            np.random.seed(42)
            indices = np.random.choice(len(all_images), size=int(0.1 * len(all_images)), replace=False)
            all_images = [all_images[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]

        assert len(all_images) == len(all_masks), (
            f"Images ({len(all_images)}) vs masks ({len(all_masks)}) mismatch"
        )

        # 90/10 split idx
        split_idx = int(0.9 * len(all_images))

        if train:
            self.images = all_images[:split_idx]
            self.masks = all_masks[:split_idx]
        else:
            self.images = all_images[split_idx:]
            self.masks = all_masks[split_idx:]

        # Build a lookup table for mapping IDs to trainIds
        self.lut = np.ones(256, dtype=np.uint8)*155  # 256 for all possible uint8 values
        for k, v in id2label.items():
            self.lut[k] = v.trainId
        # Ensure that the values of the lookup table are in perfect ascending order 0,1,2,3... except for 255 which is ignored
        self.unique_mask_values = np.unique(self.lut)
        f1 = self.unique_mask_values != 255
        f2 = self.unique_mask_values != 155
        self.unique_mask_values = self.unique_mask_values[f1 & f2]  # Ignore 255 and 155
        if not np.array_equal(self.unique_mask_values, np.arange(len(self.unique_mask_values))):
            raise ValueError("TrainIds are not in perfect ascending order.")

        # Transforms
        self.transform = transform

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

        # Convert image and mask to [0, 255] uint8 numpy arrays
        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

        # Use the lookup table to map the mask in a vectorized way
        train_mask = self.lut[mask] # ID -> trainId
        # Assert that no value is outside the unique mask values
        if np.any(train_mask == 155):
            raise ValueError("Found an invalid mask (mapping: 155).")

        if self.transform is not None:
            augmented = self.transform(image=image, mask=train_mask)
            image, train_mask = augmented['image'], augmented['mask']
            
        return image, train_mask

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
