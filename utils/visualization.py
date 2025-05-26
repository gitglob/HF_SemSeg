import torch
import matplotlib.pyplot as plt
from torchmetrics import JaccardIndex
import wandb
from data.labels_kitti360 import NUM_CLASSES


def plot_image_and_masks(image: torch.tensor, 
                         ground_truth: torch.tensor, 
                         prediction: torch.tensor, 
                         attention_map: torch.tensor, 
                         epoch: int):
    """Generate and log plots for a specific image."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image.numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground truth segmentation
    axes[1].imshow(ground_truth.numpy(), cmap="viridis")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted segmentation
    axes[2].imshow(prediction.numpy(), cmap="viridis")
    axes[2].set_title("Predicted Segmentation")
    axes[2].axis("off")

    # Attention map
    if attention_map is not None:
        axes[3].imshow(attention_map.numpy(), cmap="jet")
        axes[3].set_title("Attention Map")
        axes[3].axis("off")

    # Metric: mean IoU
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='macro',
        ignore_index=255
    )
    miou_metric.reset()
    miou_metric.update(prediction, ground_truth)
    miou = miou_metric.compute().item()

    plt.tight_layout()
    wandb.log({f"Validation Plots (Epoch {epoch}), mIoU: {miou:.4f}": wandb.Image(fig)})
    plt.close(fig)
