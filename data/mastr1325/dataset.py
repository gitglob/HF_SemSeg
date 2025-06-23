import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 4

id2label = {
    0: "obstacles and environment",
    1: "water",
    2: "sky",
    4: "unknown"
}

def get_sorted_file_list(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')]
    files.sort()
    return files

class MaritimeDataset(Dataset):
    """
    Dataset for MaSTr1325 semantic segmentation.
    Expects directory structure:
      root/
        images/         # input RGB images (.png)
        masks/          # ground-truth labels (class IDs, .png)
    """
    def __init__(self, root_dir,
                 train=True,
                 transform=None,
                 debug=False):
        self.root_dir = root_dir

        # Read the match file and extract image and mask paths
        all_images, all_masks = self._read_dataset(root_dir, debug)

        # If in debug mode, only keep a random 10% of the dataset
        if debug:
            np.random.seed(42)
            indices = np.random.choice(len(all_images), size=int(0.1 * len(all_images)), replace=False)
            all_images = [all_images[i] for i in indices]
            all_masks = [all_masks[i] for i in indices]

        assert len(all_images) == len(all_masks), (
            f"Images ({len(all_images)}) vs masks ({len(all_masks)}) mismatch"
        )

        # 80/20 split idx
        split_idx = int(0.8 * len(all_images))

        if train:
            self.images = all_images[:split_idx]
            self.masks = all_masks[:split_idx]
        else:
            self.images = all_images[split_idx:]
            self.masks = all_masks[split_idx:]

        # Build a lookup table for mapping IDs to trainIds
        self.lut = np.ones(256, dtype=np.uint8)*155  # 256 for all possible uint8 values
        for k, v in id2label.items():
            if k != 4:
                self.lut[k] = k
            else:
                self.lut[k] = 3

        # Ensure that the values of the lookup table are in perfect ascending order 0,1,2,3... except for 255 which is ignored
        self.unique_mask_values = np.unique(self.lut)
        f1 = self.unique_mask_values != 255
        f2 = self.unique_mask_values != 155
        self.unique_mask_values = self.unique_mask_values[f1 & f2]  # Ignore 255 and 155
        if not np.array_equal(self.unique_mask_values, np.arange(len(self.unique_mask_values))):
            raise ValueError("TrainIds are not in perfect ascending order.")

        # Transforms
        self.transform = transform

    def _read_dataset(self, root_dir, debug=False):
        """
        Reads the image and mask directories and extracts image and mask paths.

        Args:
            folder (str): Path to the folder containing images and masks.
        """
        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        image_files = get_sorted_file_list(img_dir)
        mask_files = get_sorted_file_list(mask_dir)

        if debug:
            return image_files[:50], mask_files[:50]
        return image_files, mask_files
    
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

    dataset_root = '/home/panos/Documents/data/mastr1325'
    transform = A.Compose([
        A.Resize(height=512,
                 width=512,
                 interpolation=cv2.INTER_LINEAR),
        ToTensorV2()
    ])
    dataset = MaritimeDataset(dataset_root, train=False, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

    for imgs, masks in loader:
        print(imgs.shape, masks.shape)
        # Take the first image and mask from the batch
        img = imgs[0].permute(1, 2, 0).cpu().numpy()
        mask = masks[0].squeeze(0).cpu().numpy()
        print(img.dtype, mask.dtype)
        
        # Denormalize and plot
        fig = plot_sample(img, mask)
        plt.show()
        break
    