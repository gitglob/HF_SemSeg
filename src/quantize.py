import time
import psutil
import numpy as np
import pynvml
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torch.quantization import quantize_dynamic
import os
import sys
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2


"""
Quantization on CPU  performance:

Original model:
    mIoU: ~ 0.81
    GPU:  ~ 0.46 sec
    CPU:  ~ 2.10 sec
Quantized model:
    mIoU: ~ 0.64
    CPU:  ~ 1.61 sec
"""

pynvml.nvmlInit()

# Add the project root directory to the Python path
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(str(project_root))

from models.DinoFPNhd import DinoFPN
from data.dataset import KittiSemSegDataset

# Hyperparameters
BATCH_SIZE = 1
NUM_CLASSES = 33

def main(cfg: DictConfig):
    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    cmap = plt.get_cmap("viridis", NUM_CLASSES)
    norm = BoundaryNorm(boundaries=np.arange(NUM_CLASSES + 1) - 0.5, ncolors=NUM_CLASSES)

    # Initialize model
    model = DinoFPN(num_labels=cfg.dataset.num_classes, model_cfg=cfg.model)

    # Load the checkpoint
    checkpoint_path = f"checkpoints/{cfg.checkpoint.model_name}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    # Quantize the model
    qmodel = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},  # Specify which layers to quantize
        dtype=torch.qint8,  # Quantization data type
        inplace=False  # Do not modify the original model
    )

    # Metric: mean IoU
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='micro',
        ignore_index=None
    )

    model.eval()
    qmodel.eval()
    running_miou = 0.0
    running_inference_time_cpu = 0.0
    running_inference_time_gpu = 0.0
    running_qmiou = 0.0
    running_qinference_time_cpu = 0.0

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(val_loader):
            masks = masks  # [B, H, W]
            imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            print(f"\nProcessing image {idx + 1}/{len(val_loader)}]")

            print("\t Running inference on the original model...")
            logits, dt = inference(model, imgs, "cpu")
            running_inference_time_cpu += dt
            logits, dt = inference(model, imgs, "cuda:0")
            running_inference_time_gpu += dt
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            miou = est_miou(miou_metric, preds, masks)
            running_miou += miou

            # Run inference on the model and estimate mIoU
            print("\t Running inference on the quantized model...")
            qlogits, dt = inference(qmodel, imgs, "cpu")
            running_qinference_time_cpu += dt
            qpreds = torch.argmax(qlogits, dim=1)  # [B, H, W]
            qmiou = est_miou(miou_metric, qpreds, masks)
            running_qmiou += qmiou

            breakpoint()
            # Log the results
            if idx % 10 == 0:
                print(f"\t ~~Original model~~ \n"
                      f"mIoU: {running_miou / (idx + 1):.4f} \n"
                      f"Inference time (CPU): {running_inference_time_cpu / (idx + 1):.4f} s \n"
                      f"Inference time (GPU): {running_inference_time_gpu / (idx + 1):.4f} s")
                print(f"\t ~~Quantized model~~ \n"
                      f"mIoU: {running_qmiou / (idx + 1):.4f} \n"
                      f"Inference time (CPU): {running_qinference_time_cpu / (idx + 1):.4f} s \n")

            # Print results
            # plot_results(imgs, masks, preds, miou, cmap, norm)

def print_process_memory():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_bytes = process.memory_info().rss       # Resident Set Size (bytes)
    print(f"   [proc] memory (RSS) = {mem_bytes/(1024**2):.1f} MiB")
    
def print_gpu_stats(gpu_index=0):
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util    = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"   [nvidia-smi] GPU{gpu_index}: util={util.gpu}% "
          f"mem_tot={mem_info.total//(1024**2)} MiB  "
          f"used={mem_info.used//(1024**2)} MiB  "
          f"free={mem_info.free//(1024**2)} MiB")
    
def print_cpu_stats():
    # CPU % over all cores
    cpu_percent = psutil.cpu_percent(interval=None)  # non-blocking snapshot
    # Total + available + used system memory
    vm = psutil.virtual_memory()
    print(f"   [CPU] util: {cpu_percent:.1f}%   "
          f"mem_total={vm.total/(1024**2):.0f} MiB  "
          f"mem_used={vm.used/(1024**2):.0f} MiB  "
          f"mem_free={vm.available/(1024**2):.0f} MiB")

def est_miou(miou_metric, preds, masks):
    # Compute mIoU for the current image
    miou_metric.reset()
    miou_metric.update(preds, masks)
    miou = miou_metric.compute().item()

    # print(f"mIoU = {miou:.4f}")
    num_identical = torch.sum(preds == masks).item()  
    # print(f"Number of identical pixels: {num_identical}")
    num_non_identical = torch.sum(preds != masks).item()
    # print(f"Number of non-identical pixels: {num_non_identical}")
    # print(f"Accuracy: {num_identical / (num_identical + num_non_identical):.4f}")

    return miou

def inference(model, imgs, device):
    print("Device:", device)
    if device == "cpu":
        print_cpu_stats()
    elif device.startswith("cuda"):
        print_gpu_stats(0)

    model = model.to(device)
    imgs = imgs.to(device)

    if device == "cpu":
        print_cpu_stats()
    elif device.startswith("cuda"):
        print_gpu_stats(0)

    # Forward pass with inference time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(imgs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    inference_time = t1 - t0

    return logits.to("cpu"), inference_time

def plot_results(imgs, masks, preds, miou, cmap, norm):
    # Plot the image, ground truth, and prediction
    img_np = imgs[0].permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    mask_np = masks[0].cpu().numpy()
    pred_np = preds[0].cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(img_np)
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(mask_np, cmap=cmap, norm=norm)
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    ax3.imshow(pred_np, cmap=cmap, norm=norm)
    ax3.set_title(f"Prediction (mIoU={miou:.4f})")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="test"
    ):
        cfg = compose(config_name="config")
        main(cfg)