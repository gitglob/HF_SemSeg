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
    def __init__(self, root_dir, image_folder='image_2', mask_folder='semantic',
                 train=True,
                 transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_folder)
        self.mask_dir = os.path.join(root_dir, mask_folder)

        all_images = get_sorted_file_list(self.image_dir)
        all_masks  = get_sorted_file_list(self.mask_dir)
        assert len(all_images) == len(all_masks), (
            f"Images ({len(all_images)}) vs masks ({len(all_masks)}) mismatch"
        )

        # 80/20 split idx
        split_idx = int(0.8 * len(all_images))

        if train:
            self.images = all_images[:split_idx]
            self.masks  = all_masks[:split_idx]
        else:
            self.images = all_images[split_idx:]
            self.masks  = all_masks[split_idx:]

        # transforms
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = np.array(image, dtype=np.uint8)
        mask = np.array(mask, dtype=np.uint8)

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
