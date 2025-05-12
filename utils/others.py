import os
import torch
import torch.nn.functional as F


def get_cls_attention_map(attentions, H, W, patch_size):
    att = attentions[-1]            # Get the attention map from the last attention layer (batch_size, num_heads, seq_len, seq_len)
    att = att.mean(dim=1)           # Average over all attention heads (batch_size, seq_len, seq_len)
    cls_map = att[0, 0, 1:]         # Get the attention of the [CLS] token to all image patches for the first image (seq_len-1)
    cls_map = cls_map.reshape(
        H // patch_size, 
        W // patch_size
    )                               # Reshape to the logit size (H/patch, W/patch)
    cls_map = cls_map.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimension (1, 1, H/patch, W/patch)
    cls_map = F.interpolate(cls_map, size=(H, W), 
                            mode="bilinear", align_corners=False) # Reshape to the original image size (H, W)
    cls_map = cls_map.squeeze(0).squeeze(0)  # Remove batch and channel dimensions (H, W)
    cls_map = cls_map.detach().cpu().numpy()

def save_checkpoint(model, optimizer, epoch, best_val_miou, checkpoint_path):
    """Save model, optimizer, epoch, and best_val_miou to a checkpoint."""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_miou": best_val_miou
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with mIoU={best_val_miou:.4f}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model, optimizer, epoch, and best_val_miou from a checkpoint."""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_miou = checkpoint["best_val_miou"]
        print(f"Checkpoint loaded: Resuming from epoch {epoch} with mIoU={best_val_miou:.4f}")
        return epoch, best_val_miou
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, 0.0

def print_model_and_gpu_stats(model, device=torch.device('cuda:0')):
    # 1) Estimate model size on GPU (parameters only, in MB)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mb    = param_bytes / 1024**2
    print(f"Model parameters: {param_mb:.2f} MB")

    if device.type == 'cuda':
        # make sure we're measuring from the right device
        torch.cuda.synchronize(device)

        # 2) GPU memory stats (in MB)
        allocated_mb = torch.cuda.memory_allocated(device)  / 1024**2
        reserved_mb  = torch.cuda.memory_reserved(device)   / 1024**2
        total_mb     = torch.cuda.get_device_properties(device).total_memory / 1024**2
        free_mb      = total_mb - reserved_mb

        print(f"CUDA total memory   : {total_mb:7.2f} MB")
        print(f"CUDA allocated      : {allocated_mb:7.2f} MB")
        print(f"CUDA reserved       : {reserved_mb:7.2f} MB")
        print(f"CUDA free (est.)    : {free_mb:7.2f} MB")
    else:
        print("Device is not CUDA, skipping GPU stats.")
        