import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import wandb


def plot_image_and_masks(image, ground_truth, prediction, attention_map, epoch, num_classes):
    """Generate and log plots for a specific image."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    cmap = plt.get_cmap("viridis", num_classes)
    norm = BoundaryNorm(boundaries=np.arange(num_classes + 1) - 0.5, ncolors=num_classes)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth segmentation
    axes[1].imshow(ground_truth, cmap=cmap, norm=norm)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted segmentation
    axes[2].imshow(prediction, cmap=cmap, norm=norm)
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis("off")

    # Attention map
    if attention_map is not None:
        axes[3].imshow(attention_map, cmap="jet")
        axes[3].set_title("Attention Map")
        axes[3].axis("off")

    plt.tight_layout()
    wandb.log({f"Validation Plots (Epoch {epoch})": wandb.Image(fig)})
    plt.close(fig)