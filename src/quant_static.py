#!/usr/bin/env python3
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.quantization as tq
import torch.nn.intrinsic as nni
from torch.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torch.quantization import QConfig
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import albumentations as A
from omegaconf import OmegaConf, DictConfig
from hydra import initialize, compose
from pathlib import Path
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# Make sure your project root is in PYTHONPATH so we can import models & datasets
cur_dir     = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(cur_dir)
sys.path.append(project_dir)

from models.DinoFPNbn import DinoFPN       # This version uses BatchNorm2d
from data.dataset import KittiSemSegDataset
from data.labels_kitti360 import NUM_CLASSES
# ────────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[QuantPTQ] Running on device = {device}")

# ────────────────────────────────────────────────────────────────────────────────
# 1) Build a wrapper around the FP32 DinoFPN so we can insert QuantStubs and DeQuantStubs.
#    Because FPNHead now uses Conv2d → BatchNorm2d → ReLU, PyTorch’s fuse_modules will handle:
#       Conv2d + BatchNorm2d + ReLU → FusedConvBnRelu
# ────────────────────────────────────────────────────────────────────────────────
class QuantDinoFPN(nn.Module):
    def __init__(self, num_labels: int, model_cfg):
        super().__init__()
        # 1a) QuantStub to quantize input activations
        self.quant = tq.QuantStub()

        # 1b) The original FP32 DinoFPN (with BatchNorm)
        self.fp32_model = DinoFPN(num_labels=num_labels, model_cfg=model_cfg)

        # 1c) DeQuantStub to convert final output back to FP32
        self.dequant = tq.DeQuantStub()

    def forward(self, images):
        # 1) Quantize input: attaches observers to measure activation ranges
        x = self.quant(images)

        # 2) Forward through original FP32 model
        logits = self.fp32_model(x)

        # 3) Dequantize output (brings quantized int8 result back to float)
        out = self.dequant(logits)
        return out

    def fuse_model(self):
        """
        Fuse Conv2d + BatchNorm2d + ReLU sequences, wherever they appear, across the entire network.
        Because your FPNHead modules use exactly that pattern, this will fuse:
            - Each proj: Conv2d → BatchNorm2d → ReLU → Dropout2d (Dropout is skipped in fusion)
            - The fuse block: Conv2d → BatchNorm2d → ReLU → Dropout2d
            - The classifier: Conv2d → BatchNorm2d → ReLU → Dropout2d → Conv2d  (we can fuse up to ReLU)
        PyTorch automatically handles fusing only the Conv-BN-ReLU parts, leaving dropout alone.
        """
        # Fuse in the head
        head = self.fp32_model.head

        # 1) Fuse each 1×1 proj: Conv → BatchNorm → ReLU
        for idx, proj in enumerate(head.projs):
            # proj is nn.Sequential([Conv2d, BatchNorm2d, ReLU, Dropout2d])
            torch.quantization.fuse_modules(proj,
                                           ["0", "1", "2"],  # fuse conv (idx 0), bn (idx 1), relu (idx 2)
                                           inplace=True)

        # 2) Fuse the fuse-block: Conv → BatchNorm → ReLU
        torch.quantization.fuse_modules(head.fuse,
                                       ["0", "1", "2"],  # indices: 0=Conv2d, 1=BatchNorm2d, 2=ReLU
                                       inplace=True)

        # 3) Fuse the first part of classifier: Conv → BatchNorm → ReLU
        #    classifier = nn.Sequential([Conv2d, BatchNorm2d, ReLU, Dropout2d, Conv2d])
        torch.quantization.fuse_modules(head.classifier,
                                       ["0", "1", "2"],  # fuse conv(0), bn(1), relu(2)
                                       inplace=True)
        # The final Conv2d (index 4) cannot be fused further, since there's no BatchNorm or ReLU after.

        print("[QuantDinoFPN] Fused all Conv2d+BatchNorm2d+ReLU sequences.")

# ────────────────────────────────────────────────────────────────────────────────
# 2) Calibration helper: run N batches of validation data through the model so that
#    the HistogramObservers collect activation stats (KL‐based).
# ────────────────────────────────────────────────────────────────────────────────
def calibrate_model(model: nn.Module, cal_loader: DataLoader):
    """
    Runs `num_calib_batches` batches through the model in eval mode, 
    showing a progress bar so you can see calibration progress. 
    """
    model.eval()
    with torch.no_grad():
        cal_bar = tqdm(cal_loader, desc="Calibrating")
        for batch_idx, (imgs, _) in enumerate(cal_bar, start=1):
            imgs = imgs.permute(0, 3, 1, 2).to(device) # [B, C, H, W]
            input = model.fp32_model.process(imgs)
            _ = model(input)  # Forward pass only: observers record histograms
    print(f"[Calibrate] Completed {batch_idx} batches of calibration.")

# ────────────────────────────────────────────────────────────────────────────────
# 3) mIoU evaluation helper
# ────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_miou(model: nn.Module, val_loader: DataLoader) -> float:
    model.eval()
    miou_metric = JaccardIndex(
        task="multiclass",
        num_classes=NUM_CLASSES,
        average="micro",
        ignore_index=255,
    )

    # DinoV2 ImageNet normalization values
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    for imgs, masks in tqdm(val_loader, desc="[Eval]"):
        imgs  = imgs.permute(0, 3, 1, 2).float()
        input = (imgs / 255.0 - mean) / std
        logits = model(input)                  # [B, num_classes, H, W]
        preds  = torch.argmax(logits, dim=1)  # [B, H, W]
        miou_metric.update(preds, masks)

    return miou_metric.compute().item()

def print_param_dtype(model):
    for name, param in model.named_parameters():
        print(f"{name} is loaded in {param.dtype}")

def print_all_modules_after_convert(model):
    """See what modules exist after convert"""
    print("\n=== All Modules After Convert ===")
    for name, module in model.named_modules():
        print(f"{name:60s} → {type(module).__name__}")

def check_quantized_weights(model):
    """Look for quantized weight storage"""
    print("\n=== Checking for Quantized Weights ===")
    
    quantized_count = 0
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Check for packed weights (quantized modules)
        if hasattr(module, '_packed_params'):
            print(f"{name:50s} → QUANTIZED (has _packed_params) - {module_type}")
            quantized_count += 1
        
        # Check for other quantized indicators
        elif any(x in module_type for x in ['Quantized', 'Packed', 'Int8']):
            print(f"{name:50s} → QUANTIZED MODULE: {module_type}")
            quantized_count += 1
        
        # Regular FP32 modules with weights
        elif hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
            print(f"{name:50s} → FP32 weight: {module.weight.dtype}")
    
    print(f"\nTotal quantized modules found: {quantized_count}")

def print_quantizable_modules(model: torch.nn.Module):
    """
    Walk through model.named_modules() and print every submodule
    whose qconfig is not None (i.e. it will be quantized).
    """
    print("\n=== Modules with qconfig (will be quantized) ===")
    for name, module in model.named_modules():
        if getattr(module, "qconfig", None) is not None:
            print(f"{name:50s} → {type(module).__name__}")
        else:
            print(f"{name:50s} → None")

def print_quantized_modules(model: torch.nn.Module):
    """
    Walk through model.named_modules() and print any module whose
    type indicates it was replaced by a quantized implementation.
    """
    quantized_prefixes = (
        "QuantizedConv2d",
        "Conv2dPackedParams",   # older PyTorch versions
        "LinearPackedParams",   # older PyTorch versions
        "QuantizedLinear",
        "QuantizedReLU",
        "Quantize",
        "DeQuantize",
        "FloatFunctional",      # e.g. for quantized add/mul
    )

    print("\n=== Quantized submodules ===")
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        for prefix in quantized_prefixes:
            if cls_name.startswith(prefix):
                print(f"{name:50s} → {cls_name}")
                break

def print_observers(model: torch.nn.Module):
    print("\n=== Weight observers in the model ===")
    for name, module in model.named_modules():
        if getattr(module, "weight_fake_quant", None) is not None:
            breakpoint()
            obs = module.weight_fake_quant
            print(
                f"{name:60s} → weight observer={type(obs).__name__}, "
                f"qscheme={obs.qscheme}"
            )
    print("\n=== Activation observers in the model ===")
    for name, module in model.named_modules():
        if getattr(module, "activation_post_process", None) is not None:
            obs = module.activation_post_process
            print(
                f"{name:60s} → activation observer={type(obs).__name__}, "
                f"qscheme={obs.qscheme}"
            )

def inspect_weight_observers(model: torch.nn.Module):
    """
    After `prepare(model, inplace=True)`, each quantizable module—
    whether a fused ConvReLU2d, an unfused Conv2d, or a Linear—will have
    either `weight_fake_quant` or `weight_post_process` pointing at an observer.
    This helper finds and prints them, without requiring you to guess the exact class.
    """
    print("\n=== Inspecting weight observers (universal search) ===")
    any_found = False

    for name, module in model.named_modules():
        # Try both possible observer attributes (PyTorch version may vary)
        obs = getattr(module, "weight_fake_quant", None)
        if obs is None:
            obs = getattr(module, "weight_post_process", None)

        if obs is not None:
            any_found = True
            # We found a weight observer here. Print module name, observer type, and qscheme.
            qscheme = obs.qscheme if hasattr(obs, "qscheme") else "(no qscheme)"
            print(f"{name:60s} → observer = {type(obs).__name__},  qscheme = {qscheme}")

            # If it has per‐channel min/max arrays, show their shape
            if hasattr(obs, "min_vals") and hasattr(obs, "max_vals"):
                print(f"    - per‐channel min_vals shape = {tuple(obs.min_vals.shape)}, "
                      f"max_vals shape = {tuple(obs.max_vals.shape)}")
            elif hasattr(obs, "min_val") and hasattr(obs, "max_val"):
                print(f"    - min_val = {obs.min_val:.4f}, max_val = {obs.max_val:.4f}")

            # If it's a HistogramObserver, you can peek at its histogram (optional)
            if hasattr(obs, "histogram") and hasattr(obs, "edges"):
                print(f"    - histogram bins = {tuple(obs.histogram.shape)}, "
                      f"edges = {tuple(obs.edges.shape)}")

    if not any_found:
        print(">>> No weight observers found. Are you sure you called `prepare(...)` on this exact model?")

def debug_quantization_setup(model: torch.nn.Module):
    """Debug what's happening with quantization setup"""
    print("\n=== Debugging Quantization Setup ===")
    
    # Check if model has any qconfig attributes
    has_qconfig = hasattr(model, 'qconfig') and model.qconfig is not None
    print(f"Model has qconfig: {has_qconfig}")
    if has_qconfig:
        print(f"Model qconfig: {model.qconfig}")
    
    # Check all modules and their attributes
    quantizable_modules = []
    prepared_modules = []
    
    for name, module in model.named_modules():
        # Check for qconfig
        module_qconfig = getattr(module, 'qconfig', None)
        
        # Check for observer attributes
        has_weight_obs = (hasattr(module, 'weight_fake_quant') or 
                         hasattr(module, 'weight_post_process'))
        has_activation_obs = (hasattr(module, 'activation_post_process') or
                            hasattr(module, 'activation_fake_quant'))
        
        # Check if it's a standard quantizable type
        is_quantizable_type = isinstance(module, (
            torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
            torch.nn.Linear, torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d
        ))
        
        if is_quantizable_type:
            quantizable_modules.append(name)
            
        if has_weight_obs or has_activation_obs or module_qconfig:
            prepared_modules.append((name, type(module).__name__, 
                                   has_weight_obs, has_activation_obs, module_qconfig))
    
    print(f"\nFound {len(quantizable_modules)} quantizable modules:")
    for name in quantizable_modules[:10]:  # Show first 10
        print(f"  - {name}")
    if len(quantizable_modules) > 10:
        print(f"  ... and {len(quantizable_modules) - 10} more")
    
    print(f"\nFound {len(prepared_modules)} prepared modules:")
    for name, mod_type, has_w, has_a, qconf in prepared_modules:
        print(f"  - {name:40s} ({mod_type:15s}) weight_obs={has_w}, act_obs={has_a}, qconfig={qconf is not None}")

    return len(prepared_modules) > 0

def check_fused_modules_observers(model):
    """Check if fused modules have the right observers"""
    print("\n=== Checking Fused Modules ===")
    
    for name, module in model.named_modules():
        if 'ConvReLU2d' in str(type(module)):
            print(f"{name:50s} → {type(module).__name__}")
            
            # Check if it has weight observer
            weight_obs = getattr(module, 'weight_fake_quant', None)
            if weight_obs is None:
                weight_obs = getattr(module, 'weight_post_process', None)
            
            print(f"  - Has weight observer: {weight_obs is not None}")
            if weight_obs:
                print(f"  - Weight observer type: {type(weight_obs).__name__}")
            
            # Check qconfig
            qconfig = getattr(module, 'qconfig', None)
            print(f"  - Has qconfig: {qconfig is not None}")

# ────────────────────────────────────────────────────────────────────────────────
# 4) Main script: load FP32 checkpoint → wrap in QuantDinoFPN → fuse → set qconfig
#    → prepare → calibrate → convert → evaluate → save INT8 weights
# ────────────────────────────────────────────────────────────────────────────────
def main(cfg: DictConfig):
    # 4a) Build a validation loader (no augmentations, just center-crop)
    crop_h, crop_w = cfg.augmentation.crop_height, cfg.augmentation.crop_width

    val_transform = A.Compose([
        A.CenterCrop(crop_h, crop_w)
    ])

    val_dataset = KittiSemSegDataset(
        root_dir='/home/panos/Documents/data/kitti-360',
        train=False,
        transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    print(f"[QuantPTQ] Validation dataset size: {len(val_dataset)}")

    cal_dataset = KittiSemSegDataset(
        root_dir='/home/panos/Documents/data/kitti-360',
        train=True,
        calibration=True,
        transform=val_transform
    )
    cal_loader = DataLoader(
        cal_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    print(f"[QuantPTQ] Calibration dataset size: {len(cal_dataset)}")

    # 4b) Instantiate QuantDinoFPN
    quant_model = QuantDinoFPN(num_labels=cfg.dataset.num_classes, model_cfg=cfg.model)
    quant_model.to(torch.device("cpu"))
    print("[QuantPTQ] Created QuantDinoFPN.")

    # 4c) Load your best FP32 checkpoint into `quant_model.fp32_model`
    ckpt_path = os.path.join(project_dir, "checkpoints", f"{cfg.checkpoint.model_name}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    quant_model.fp32_model.load_state_dict(checkpoint["model_state_dict"])
    quant_model.eval()
    print(f"[QuantPTQ] Loaded FP32 weights from {ckpt_path} into fp32_model.")

    # 4d) Fuse Conv2d + BatchNorm2d + ReLU sequences in the head
    quant_model.fuse_model()

    # 4e) Assign QConfig: per‐channel weight quant + HistogramObserver (KL) activations
    #     - get_default_qconfig("fbgemm") already uses PerChannelMinMaxObserver for weights
    #     - We swap the activation observer to HistogramObserver (KL)
    # - Build a per-channel symmetric observer for weights
    weight_obs  = PerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )

    # - Build a histogram (KL) observer for activations
    activation_obs = HistogramObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False
    )

    # - Compose them into a QConfig
    custom_qconfig = QConfig(
        activation=activation_obs,
        weight=weight_obs
    )
    quant_model.qconfig = custom_qconfig
    # quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print("[QuantPTQ] Assigned QConfig: PerChannelMinMaxObserver(w) + HistogramObserver(a).")

    # 4f) Prepare for static quantization (inserts observers)
    dev = next(quant_model.parameters()).device
    assert dev.type == "cpu"
    tq.prepare(quant_model, inplace=True)
    # check_fused_modules_observers(quant_model)
    # inspect_weight_observers(quant_model)
    # debug_quantization_setup(quant_model)
    # print_quantizable_modules(quant_model)
    # print_observers(quant_model)
    print("[QuantPTQ] Model prepared for static quant (observers inserted).")

    # 4g) Calibrate on ~500 validation batches
    calibrate_model(quant_model, cal_loader)

    # 4h) Convert to INT8 (replace FP32 modules with quantized kernels)
    quant_model.eval()
    print_param_dtype(quant_model)
    
    tq.convert(quant_model, inplace=True)
    print("[QuantPTQ] Model converted to INT8 (quantized kernels).")
    # print_quantized_modules(quant_model)

    # 4i) Evaluate mIoU of the INT8 model
    int8_miou = evaluate_miou(quant_model, val_loader)
    print(f"[QuantPTQ] INT8‐quantized model mIoU = {int8_miou:.4f}")

    # 4j) Save the INT8 state_dict
    save_path = os.path.join(project_dir, "checkpoints", "dinofpn_int8.pth")
    torch.save(quant_model.state_dict(), save_path)
    print(f"[QuantPTQ] Saved INT8 weights to {save_path}")

    return

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with initialize(version_base=None, config_path=f"../configs", job_name="quant_static_ptq"):
        cfg = compose(config_name="quant_config")
        main(cfg)
