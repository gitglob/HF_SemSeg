#!/usr/bin/env python3
import time
import psutil
import csv
import os
import sys
from pathlib import Path

import torch
import pynvml
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex

import torch.quantization as tq
from torch.quantization import quantize_dynamic
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.quantize_fx import convert_fx
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import PerChannelMinMaxObserver, HistogramObserver, MinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as LearnableFakeQuantize
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2PatchEmbeddings, Dinov2Embeddings

from hydra import compose, initialize
from omegaconf import DictConfig

# Initialize NVML for GPU queries
pynvml.nvmlInit()

# Add project root to path (so that `models` and `data` can be imported)
cur_dir = Path(__file__).parent
project_dir = cur_dir.parent
sys.path.append(str(project_dir))

from models.DinoFPNbn import DinoFPN
from data.kitti360.dataset import KittiSemSegDataset
from utils.others import get_memory_footprint, get_quant_memory_footprint


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
Results - Quantization on CPU performance:

Original model:
    Size: 90,328,097 params, 344.57 MB
    mIoU: ~ 0.85
    GPU:
        Inference time: ~ 0.46 sec
        Power draw:     ~ 20.0 J
    CPU:
        Inference time: ~ 1.9 sec
        Power draw:     ~ 90 J

Quantized model (Dynamic INT8):
    Size: (didn't compute size, should be close to the QAT model)
    mIoU: ~ 0.79
    CPU:
        Inference time: ~ 1.4 sec
        Power draw:     ~ 67 J

QAT Quantized model (INT8):
    Size: 94,686,819 params, 108.13 MB
    mIoU: ~ 0.84
    CPU:
        Inference time: ~ 1.4 sec
        Power draw:     ~ 67 J
"""

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

def create_qat_model_structure(fp32_model):
    """Create and prepare a QAT model identical to the notebook"""    
    # Build MinMax observer for weights
    ch_qat_weight = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0
    )
    tensor_qat_weight = MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
    )

    # - Build a histogram (KL) observer for activations
    activation_obs = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )
    
    qat_activation = FakeQuantize.with_args(
        observer=activation_obs
    )

    custom_qconfig = QConfig(
        activation=qat_activation, 
        weight=tq.default_fused_per_channel_wt_fake_quant
    )

    l0_qconfig = QConfig(
        activation=qat_activation, 
        weight=tensor_qat_weight
    )
    head_qconfig = QConfig(
        activation=qat_activation, 
        weight=ch_qat_weight
    )

    qconfig_map = (
        QConfigMapping()
        .set_global(custom_qconfig)                   # applies to all modules by default
        .set_module_name("backbone.embeddings", None)  # disable quant for embeddings
        # .set_module_name("backbone.embeddings.patch_embeddings", None)
        # CRITICAL LAYERS: Use learnable quantization
        .set_module_name("backbone.encoder.layer.0.attention.query", l0_qconfig)
        .set_module_name("backbone.encoder.layer.0.attention.key", l0_qconfig)
        .set_module_name("backbone.encoder.layer.0.attention.value", l0_qconfig)
        .set_module_name("backbone.head.classifier", head_qconfig)
        # Exclude normalization layers
        .set_object_type(torch.nn.LayerNorm, None)
        .set_object_type(torch.nn.BatchNorm2d, None)
        .set_object_type(torch.nn.GroupNorm, None)
        # Exclude dropout layers
        .set_object_type(torch.nn.Dropout, None)
        # Exclude operations that return non-tensor objects
        .set_object_type("size", None)
        .set_object_type("view", None)
        .set_object_type("reshape", None)
        .set_object_type("permute", None)
        .set_object_type(torch.Tensor.size, None)
        .set_object_type(torch.Tensor.view, None)
        .set_object_type(torch.Tensor.reshape, None)
        .set_object_type(torch.Tensor.permute, None)
    )
    print("QConfig mapping created.")
    
    # Prepare custom config (identical to notebook)
    prepare_custom_config = (
        PrepareCustomConfig()
        .set_non_traceable_module_classes([
            Dinov2PatchEmbeddings, 
            Dinov2Embeddings
        ])
        # Exclude tensor operations that don't need quantization
        .set_preserved_attributes([
            "size", "view", "reshape", "permute", 
            "transpose", "contiguous", "flatten"
        ])
    )
    print("Custom prepare config set for non-traceable modules.")
    
    # Prepare model for QAT
    example_inputs = (torch.randn(1, 3, 364, 1232),)
    prep_model = prepare_qat_fx(
        fp32_model, 
        qconfig_map, 
        example_inputs,
        prepare_custom_config=prepare_custom_config
    )    
    print(f"QAT model prepared")
    
    return prep_model

def create_qat_model(cfg):
    # Build the FP32 model
    fp32_model = DinoFPN(
        num_labels=cfg.dataset.num_classes, 
        model_cfg=cfg.model
    )
    print("Loaded FP32 model.")

    # Build MinMax observer for weights
    learnable_ch_qat_weight = LearnableFakeQuantize.with_args(
        observer=PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0
        ),
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    learnable_tensor_qat_weight = LearnableFakeQuantize.with_args(
        observer=MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False
        ),
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        scale=1.0,        
        zero_point=0.0    
    )

    # - Build a histogram (KL) observer for activations
    activation_obs = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )

    learnable_qat_activation = LearnableFakeQuantize.with_args(
        observer=activation_obs,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
        scale=1.0,
        zero_point=128.0
    )
    qat_activation = FakeQuantize.with_args(
        observer=activation_obs
    )

    custom_qconfig = QConfig(
        activation=qat_activation, 
        weight=tq.default_fused_per_channel_wt_fake_quant
    )
    learnable_l0_qconfig = QConfig(
        activation=learnable_qat_activation, 
        weight=learnable_tensor_qat_weight
    )
    learnable_head_qconfig = QConfig(
        activation=learnable_qat_activation, 
        weight=learnable_ch_qat_weight
    )

    qconfig_map = (
        QConfigMapping()
        .set_global(custom_qconfig)                   # applies to all modules by default
        .set_module_name("backbone.embeddings", None)  # disable quant for embeddings
        # .set_module_name("backbone.embeddings.patch_embeddings", None)
        # CRITICAL LAYERS: Use learnable quantization
        .set_module_name("backbone.encoder.layer.0.attention.query", learnable_l0_qconfig)
        .set_module_name("backbone.encoder.layer.0.attention.key", learnable_l0_qconfig)
        .set_module_name("backbone.encoder.layer.0.attention.value", learnable_l0_qconfig)
        .set_module_name("backbone.head.classifier", learnable_head_qconfig)
        # Exclude normalization layers
        .set_object_type(torch.nn.LayerNorm, None)
        .set_object_type(torch.nn.BatchNorm2d, None)
        .set_object_type(torch.nn.GroupNorm, None)
        # Exclude dropout layers
        .set_object_type(torch.nn.Dropout, None)
        # Exclude operations that return non-tensor objects
        .set_object_type("size", None)
        .set_object_type("view", None)
        .set_object_type("reshape", None)
        .set_object_type("permute", None)
        .set_object_type(torch.Tensor.size, None)
        .set_object_type(torch.Tensor.view, None)
        .set_object_type(torch.Tensor.reshape, None)
        .set_object_type(torch.Tensor.permute, None)
    )
    print("QConfig mapping created.")


    # Tell FX to treat Dinov2PatchEmbeddings and Dinov2Embeddings as non-traceable
    ## “Whenever you hit SomeModule in the model, don’t open it up and record its internal steps. 
    ## Instead just treat the whole call as a single step in the recipe
    prepare_custom_config = (
        PrepareCustomConfig()
        .set_non_traceable_module_classes([
            Dinov2PatchEmbeddings, 
            Dinov2Embeddings
        ])
        # Exclude tensor operations that don't need quantization
        .set_preserved_attributes([
            "size", "view", "reshape", "permute", 
            "transpose", "contiguous", "flatten"
        ])
    )
    print("Custom prepare config set for non-traceable modules.")

    # Load your best FP32 checkpoint into `quant_model.fp32_model`
    ckpt_path = os.path.join(project_dir, "checkpoints", f"{cfg.checkpoint.model_name}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    fp32_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded FP32 weights from {ckpt_path} into fp32_model.")

    # Prepare the model for static quantization
    example_inputs = (torch.randn(1, 3, 364, 1232),)
    prep_model = prepare_qat_fx(
        fp32_model, 
        qconfig_map, 
        example_inputs,
        prepare_custom_config=prepare_custom_config
    )
    prep_model = prep_model.to("cpu")
    print("Prepared model for QAT.")

    return prep_model

def load_qat_model(model_path, fp32_model):
    """Load a saved quantized INT8 model"""    
    # First create the QAT model
    prep_model = create_qat_model_structure(fp32_model)  # Always prepare on CPU
    
    # Load the saved quantized weights
    if Path(model_path).exists():
        state_dict = torch.load(model_path)
        prep_model.load_state_dict(state_dict)
        print(f"Loaded quantized model from {model_path}")
    else:
        print(f"Warning: {model_path} not found, returning freshly converted model")
    
    # Convert to quantized model
    prep_model.eval()
    prep_model = prep_model.to("cpu")
    quant_model = convert_fx(prep_model)
    
    return quant_model

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
    checkpoint_path = project_dir / "checkpoints" / "dino-fpn-bn.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded original model from {checkpoint_path}")
    get_memory_footprint(model)

    # Create quantized model (dynamic quantization)
    qdyn_model = quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8,
        inplace=False
    )
    qdyn_model.eval()
    print("Created quantized version of the model (INT8 dynamic).")

    # Create QAT quantized model
    model_path = project_dir / "checkpoints" / f"dino-fpn-qat-int8.pth"
    qat_model = load_qat_model(model_path, model)
    print("Created QAT quantized version of the model (INT8).")
    get_quant_memory_footprint(qat_model)

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
                imgs = model.process(imgs)

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

                # 3) Dynamically Quantized model on CPU
                row = {"idx": idx, "model_type": "dyn-quantized", "device": "cpu"}
                metrics = run_inference_and_log(qdyn_model, imgs, masks, "cpu", miou_metric)
                row.update(metrics)
                writer.writerow(row)

                # 4) QAT Quantized model on CPU
                row = {"idx": idx, "model_type": "qat-quantized", "device": "cpu"}
                metrics = run_inference_and_log(qat_model, imgs, masks, "cpu", miou_metric)
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
