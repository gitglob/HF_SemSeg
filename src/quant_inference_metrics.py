#!/usr/bin/env python3
import time
import psutil
import csv
import os
import sys
from pathlib import Path

import torch
import numpy as np
import pynvml
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from torch.quantization import quantize_dynamic

from hydra import compose, initialize
from omegaconf import DictConfig
from albumentations.pytorch import ToTensorV2

# ────────────────────────────────────────────────────────────────────────────────
# This script runs inference on both the original and quantized DinoFPN model
# over the entire validation dataloader, and logs timing + resource metrics
# to a CSV file. Metrics include:
#   • Inference time (s)
#   • CPU utilization (%) and system-RAM usage (MiB)
#   • Process-RAM usage (RSS in MiB)
#   • GPU utilization (%) and GPU‐RAM usage (MiB) when running on CUDA
#   • mIoU per inference
#
# Usage:
#   $ python quant_inference_metrics.py
#
# Remember to run "sudo chmod o+r /sys/class/powercap/intel-rapl:*/*/energy_uj" before running this script
# ────────────────────────────────────────────────────────────────────────────────

"""
Results:
Quantization on CPU performance:
Original model:
    mIoU: ~ 0.81
    GPU:
        Inference time: ~ 0.46 sec
        Power draw:     ~ 20.0 J
    CPU:
        Inference time: ~ 1.90 sec
        Power draw:     ~ 90 J
Quantized model:
    mIoU: ~ 0.61
    CPU:
        Inference time: ~ 1.45 sec
        Power draw:     ~ 67 J
"""


# Initialize NVML for GPU queries
pynvml.nvmlInit()

# Add project root to path (so that `models` and `data` can be imported)
cur_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cur_dir)
sys.path.append(project_root)

from models.DinoFPNhd import DinoFPN
from data.dataset import KittiSemSegDataset

# ────────────────────────────────────────────────────────────────────────────────
# Hyperparameters
BATCH_SIZE = 1
NUM_CLASSES = 33
RAPL_PATH     = "/sys/class/powercap/intel-rapl:0/energy_uj"  # RAPL path for CPU energy

# ────────────────────────────────────────────────────────────────────────────────

def read_cpu_energy_uj():
    """
    Reads CPU energy in microjoules from RAPL.
    Returns an int (µJ) or None if the file isn't accessible.
    """
    try:
        with open(RAPL_PATH, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None

def get_gpu_power_w(gpu_index=0):
    """
    Returns instantaneous GPU power draw in Watts (float).
    NVML returns mW, so divide by 1000.
    """
    handle   = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    return power_mw / 1000.0  # convert to watts

def get_cpu_stats():
    """
    Returns: (cpu_percent, ram_used_mib, ram_free_mib)
    """
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    ram_used = vm.used // (1024**2)
    ram_free = vm.available // (1024**2)
    return cpu_percent, ram_used, ram_free

def get_proc_mem():
    """
    Returns the Python process's RSS memory in MiB.
    """
    pid = os.getpid()
    proc = psutil.Process(pid)
    rss = proc.memory_info().rss // (1024**2)
    return rss

def get_gpu_stats(gpu_index=0):
    """
    Returns: (gpu_util_percent, gpu_mem_used_mib, gpu_mem_free_mib)
    """
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util    = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_used = mem_info.used // (1024**2)
    gpu_free = mem_info.free // (1024**2)
    gpu_util = util.gpu
    return gpu_util, gpu_used, gpu_free

# ────────────────────────────────────────────────────────────────────────────────
def est_miou(miou_metric, preds, masks):
    miou_metric.reset()
    miou_metric.update(preds, masks)
    return miou_metric.compute().item()

# ────────────────────────────────────────────────────────────────────────────────
def run_inference_and_log(model, imgs, masks, device, miou_metric):
    """
    Runs a single inference pass (moving model+input to `device`),
    measures timing + resource metrics, computes mIoU,
    and returns a dict of all logged values.
    """
    # Move model + images to device
    model = model.to(device)
    imgs = imgs.to(device)

    # Read CPU energy before (µJ)
    cpu_energy_before = read_cpu_energy_uj()

    # Record metrics *before* forward pass
    cpu_before, ram_used_before, ram_free_before = get_cpu_stats()
    proc_mem_before = get_proc_mem()
    if device.startswith("cuda"):
        gpu_util_before, gpu_used_before, gpu_free_before = get_gpu_stats(int(device.split(":")[1]))
        gpu_power_before = get_gpu_power_w(int(device.split(":")[1]))
    else:
        gpu_util_before, gpu_used_before, gpu_free_before, gpu_power_before = (None, None, None, None)

    # Synchronize & run forward
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    logits = model(imgs)
    if device.startswith("cuda"): torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    inference_time = t1 - t0

    # Read CPU energy after (µJ)
    cpu_energy_after = read_cpu_energy_uj()

    # Record metrics *after* forward pass
    cpu_after, ram_used_after, ram_free_after = get_cpu_stats()
    proc_mem_after = get_proc_mem()
    if device.startswith("cuda"):
        gpu_util_after, gpu_used_after, gpu_free_after = get_gpu_stats(int(device.split(":")[1]))
        gpu_power_after = get_gpu_power_w(int(device.split(":")[1]))
    else:
        gpu_util_after, gpu_used_after, gpu_free_after, gpu_power_after = (None, None, None, None)

    # Compute CPU energy delta (in Joules)
    if cpu_energy_before is not None and cpu_energy_after is not None:
        delta_cpu_j = (cpu_energy_after - cpu_energy_before) / 1e6  # µJ → J
    else:
        delta_cpu_j = None

    # Estimate GPU energy consumed (W * seconds = J)
    if gpu_power_before is not None and gpu_power_after is not None:
        avg_gpu_power = 0.5 * (gpu_power_before + gpu_power_after)
        delta_gpu_j   = avg_gpu_power * inference_time
    else:
        delta_gpu_j = None

    # Compute preds + mIoU
    preds = torch.argmax(logits, dim=1).cpu()
    miou = est_miou(miou_metric, preds, masks)

    # Collect everything in a dict
    return {
        "inference_time_s":           inference_time,
        "miou":                       miou,
        "cpu_energy_delta_j":         delta_cpu_j,
        "cpu_energy_before_uj":       cpu_energy_before,
        "cpu_energy_after_uj":        cpu_energy_after,
        "cpu_util_before_pct":        cpu_before,
        "cpu_util_after_pct":         cpu_after,
        "ram_used_before_mib":        ram_used_before,
        "ram_used_after_mib":         ram_used_after,
        "ram_free_before_mib":        ram_free_before,
        "ram_free_after_mib":         ram_free_after,
        "proc_mem_before_mib":        proc_mem_before,
        "proc_mem_after_mib":         proc_mem_after,
        "gpu_power_before_w":         gpu_power_before,
        "gpu_power_after_w":          gpu_power_after,
        "gpu_energy_delta_j":         delta_gpu_j,
        "gpu_util_before_pct":        gpu_util_before,
        "gpu_util_after_pct":         gpu_util_after,
        "gpu_mem_used_before_mib":    gpu_used_before,
        "gpu_mem_used_after_mib":     gpu_used_after,
        "gpu_mem_free_before_mib":    gpu_free_before,
        "gpu_mem_free_after_mib":     gpu_free_after,
    }

# ────────────────────────────────────────────────────────────────────────────────
def main(cfg: DictConfig):
    # Dataset + DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=None)
    val_loader  = DataLoader(val_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)

    # Initialize original model & load checkpoint
    model = DinoFPN(num_labels=cfg.dataset.num_classes, model_cfg=cfg.model)
    checkpoint_path = f"checkpoints/{cfg.checkpoint.model_name}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded original model from {checkpoint_path}")

    # Create quantized model (dynamic quantization)
    qmodel = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
        inplace=False
    )
    qmodel.eval()
    print("Created quantized version of the model (INT8 dynamic).")

    # Prepare mIoU metric
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=NUM_CLASSES,
        average='micro',
        ignore_index=None
    )

    # Prime psutil.cpu_percent() to get a baseline
    psutil.cpu_percent(interval=None)

    # Prepare CSV logging
    project_dir = Path(project_root)
    csv_path = project_dir / "logs" / "inference_metrics_log.csv"
    header = [
        "idx",
        "model_type",
        "device",
        "inference_time_s",
        "miou",
        "cpu_energy_delta_j",
        "gpu_energy_delta_j",
        "cpu_energy_before_uj",
        "cpu_energy_after_uj",
        "cpu_util_before_pct",
        "cpu_util_after_pct",
        "ram_used_before_mib",
        "ram_used_after_mib",
        "ram_free_before_mib",
        "ram_free_after_mib",
        "proc_mem_before_mib",
        "proc_mem_after_mib",
        "gpu_power_before_w",
        "gpu_power_after_w",
        "gpu_util_before_pct",
        "gpu_util_after_pct",
        "gpu_mem_used_before_mib",
        "gpu_mem_used_after_mib",
        "gpu_mem_free_before_mib",
        "gpu_mem_free_after_mib",
    ]

    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        with torch.no_grad():
            for idx, (imgs, masks) in enumerate(val_loader):
                # Convert imgs from [B, H, W, C] → [B, C, H, W]
                imgs = imgs.permute(0, 3, 1, 2)  # shape: [1, 3, H, W]

                print(f"\n=== Processing sample {idx + 1}/{len(val_loader)} ===")

                # 1) Original model on CPU
                row = {"idx": idx, "model_type": "original", "device": "cpu"}
                metrics = run_inference_and_log(model, imgs, masks, "cpu", miou_metric)
                row.update(metrics)
                writer.writerow(row)

                # 2) Original model on GPU (if available)
                if torch.cuda.is_available():
                    row = {"idx": idx, "model_type": "original", "device": "cuda:0"}
                    metrics = run_inference_and_log(model, imgs, masks, "cuda:0", miou_metric)
                    row.update(metrics)
                    writer.writerow(row)
                else:
                    print("Skipping GPU run (no CUDA available).")

                # 3) Quantized model on CPU
                row = {"idx": idx, "model_type": "quantized", "device": "cpu"}
                metrics = run_inference_and_log(qmodel, imgs, masks, "cpu", miou_metric)
                row.update(metrics)
                writer.writerow(row)

                # Optional: break early for debugging
                # if idx >= 10:
                #     break

    print(f"\nAll inference runs complete. Metrics logged to: {csv_path}")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Compose Hydra configuration (adjust `config_path` as needed)
    with initialize(version_base=None, config_path=f"../configs", job_name="inference_metrics"):
        cfg = compose(config_name="config")
        main(cfg)
