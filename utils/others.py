import os
import torch
import torch.nn.functional as F


def get_quant_memory_footprint(model):
    """Calculate memory footprint for quantized models"""
    
    # Iterate over the model's modules
    total_params = 0
    total_bytes = 0
    for name, module in model.named_modules():
        # print(f"Module: {name}, Type: {type(module)}")
        
        if 'quantized' in str(type(module)).lower() and hasattr(module, 'weight') and callable(module.weight):
            # print("\t (quantized)")

            # Extract the weight
            w = module.weight()
            b = module.bias() if module.bias is not None else None

            # Extract the weight and bias parameters
            weight_bytes = w.numel() * w.element_size()
            bias_bytes = b.numel() * b.element_size() if b is not None else 0

            # Extract the scale and zero point parameters
            scale = w.q_per_channel_scales() if hasattr(w, 'q_per_channel_scales') else w.q_scale
            zero_point = w.q_per_channel_zero_points() if hasattr(w, 'q_per_channel_zero_points') else w.q_zero_point

            # Extract the scale and zero point sizes
            scale_bytes = scale.numel() * scale.element_size() if scale is not None else 0
            zero_point_bytes = zero_point.numel() * zero_point.element_size() if zero_point is not None else 0

            # Calculate total size in bytes
            bytes = weight_bytes + bias_bytes + scale_bytes + zero_point_bytes
            params = w.numel() + (b.numel() if b is not None else 0) + \
                    (scale.numel() if scale is not None else 0) + \
                    (zero_point.numel() if zero_point is not None else 0)

            # print(f"Module: {name}, Params: {params}, Size: {bytes / (1024**2):.2f} MB")

            # Add to total bytes and param count
            total_bytes += bytes
            total_params += params
        else:
            # print("\t (not quantized)")

            # If not quantized, just count the parameters
            params = sum(p.numel() for p in module.parameters())
            bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            # print(f"Module: {name}, Params: {params}, Size: {bytes / (1024**2):.2f} MB")

            # Add to total bytes
            total_bytes += bytes
            total_params += params

    print(f"=== Model Memory Footprint ===")
    print(f"Total:     {total_params:,} params, {total_bytes / (1024**2):.2f} MB")
    
    return total_bytes

def get_memory_footprint(model, detailed=False):
    def get_module_size(module):
        """Helper to get size of a specific module"""
        total_bytes = 0
        for param in module.parameters():
            total_bytes += param.numel() * param.element_size() # number of elements * size of each element in bytes
        return total_bytes
    
    # Get sizes for each major component
    backbone_bytes = get_module_size(model.backbone)
    head_bytes = get_module_size(model.head)
    total_bytes = backbone_bytes + head_bytes
    
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    
    print(f"=== Model Memory Footprint ===")
    if detailed:
        print(f"Backbone: {backbone_params:,} params, {backbone_bytes / (1024**2):.2f} MB")
        print(f"Head:     {head_params:,} params, {head_bytes / (1024**2):.2f} MB")
    print(f"Total:    {backbone_params + head_params:,} params, {total_bytes / (1024**2):.2f} MB")
    
    return total_bytes

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
                            mode="bilinear", align_corners=False) # Reshape to the original image size (1, 1, H, W)
    cls_map = cls_map.squeeze(0).squeeze(0)  # Remove batch and channel dimensions (H, W)
    cls_map = cls_map.detach().cpu().numpy()

    return cls_map

def save_checkpoint(model, optimizer, epoch, best_val_miou, checkpoint_cfg, scheduler):
    """Save model, optimizer, epoch, and best_val_miou to a checkpoint."""
    checkpoint_path = f"checkpoints/" + checkpoint_cfg.model_name + ".pth"
    torch.save({
        "model_name": checkpoint_cfg.model_name,
        "comment": checkpoint_cfg.comment,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_miou": best_val_miou
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} with mIoU={best_val_miou:.4f} and lr={scheduler.get_last_lr()[0]:.6f}")

def load_checkpoint(model, optimizer=None, checkpoint_cfg=None, scheduler=None):
    """Load model, optimizer, epoch, and best_val_miou from a checkpoint."""
    checkpoint_path = f"checkpoints/" + checkpoint_cfg.model_name + ".pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # if scheduler is not None:
        #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        best_val_miou = checkpoint["best_val_miou"]
        print(f"Checkpoint loaded: Resuming from epoch {epoch} with mIoU={best_val_miou:.4f} and lr={scheduler.get_last_lr()[0]:.6f}")
        return epoch, best_val_miou
    else:
        print("No checkpoint found. Starting from scratch.")
        return 1, 0.0

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
        