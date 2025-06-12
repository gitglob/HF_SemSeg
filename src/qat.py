import os
import sys
import gc
import operator
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
import albumentations as A
from hydra import initialize, compose
from pathlib import Path
from tqdm import tqdm
from transformers.models.dinov2.modeling_dinov2 import Dinov2PatchEmbeddings, Dinov2Embeddings

from torch.ao.quantization import get_default_qconfig, QConfig
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.observer import PerChannelMinMaxObserver, HistogramObserver, MinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as LearnableFakeQuantize
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig

cur_dir     = Path(__file__).parent
project_dir = cur_dir.parent
sys.path.append(str(project_dir))

from models.DinoFPNbn import DinoFPN
from data.dataset import KittiSemSegDataset
from data.labels_kitti360 import NUM_CLASSES
from utils.others import get_memory_footprint, get_quant_memory_footprint

# ────────────────────────────────────────────────────────────────────────────────

"""
Achieves 83.94% mIoU on KITTI-360 validation set with QAT
"""


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("CUDA is available.")
else:
    print("CUDA is not available, using CPU.")

def qevaluate(model, fp32_model, val_loader):
    model_cpu = deepcopy(model).to("cpu")
    print(f"QAT Model moved to CPU")
    model_cpu.eval()
    quant_model = convert_fx(model_cpu)

    # Evaluate mIoU of the INT8 model
    with torch.no_grad():
        miou_metric = JaccardIndex(
            task="multiclass",
            num_classes=NUM_CLASSES,
            average="micro",
            ignore_index=255,
        )
        val_bar = tqdm(val_loader, desc=f"[QAT] Eval")
        for batch_idx, (imgs, masks) in enumerate(val_bar, start=1):
            imgs  = imgs.permute(0, 3, 1, 2).float()
            input = fp32_model.process(imgs)
            logits = quant_model(input)           # [B, num_classes, H, W]
            preds  = torch.argmax(logits, dim=1)  # [B, H, W]
            miou_metric.update(preds, masks)

        int8_miou = miou_metric.compute().item()

    print(f"[QAT] INT8‐quantized model mIoU = {int8_miou:.4f}")
    return int8_miou

def main(cfg):
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
    prep_model = prep_model.to(device)
    print("Prepared model for QAT.")

    # Load the checkpoint if it exists
    checkpoint_path = f"checkpoints/" + cfg.checkpoint.quant_model_name + ".pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading QAT checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)
        prep_model.load_state_dict(state_dict)

    # Build a validation/calibration loader (no augmentations, just center-crop)
    crop_h, crop_w = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)

    qat_transform = A.Compose([
        A.CenterCrop(crop_h, crop_w)
    ])

    val_dataset = KittiSemSegDataset(
        root_dir='/home/panos/Documents/data/kitti-360',
        train=False,
        transform=qat_transform
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    print(f"Validation dataset size: {len(val_dataset)}")

    train_dataset = KittiSemSegDataset(
        root_dir='/home/panos/Documents/data/kitti-360',
        train=True,
        transform=qat_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )
    print(f"Train dataset size: {len(train_dataset)}")

    # Setup training components
    # criterion = CombinedLoss(alpha=0.8, ignore_index=255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(prep_model.parameters(), lr=cfg.train.learning_rate)  # Lower learning rate for QAT
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='micro',
        ignore_index=255
    ).to(device)

    # QAT Training Loop
    num_epochs = cfg.train.num_epochs  # Usually 1-5 epochs for QAT fine-tuning
    batches_per_epoch = cfg.train.batches_per_epoch
    best_val_miou = 0.0
    print(f"Starting QAT training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        print(f"QAT model moved to {device}")

        ####### TRAIN #######
        prep_model.train()
        miou_metric.reset()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"[QAT Epoch {epoch+1}/{num_epochs}] Train")
        for batch_idx, (imgs, masks) in enumerate(train_bar, start=1):
            if batch_idx >= batches_per_epoch:
                break
            imgs = imgs.permute(0, 3, 1, 2)
            
            # Forward pass through preprocessing
            input = fp32_model.process(imgs).to(device)  # [B, 3, H, W]
            
            # Forward pass through QAT model
            logits = prep_model(input)
            
            # Compute loss
            masks = masks.to(device)
            loss = criterion(logits, masks.long())
            running_train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # compute IoU
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            miou_metric.update(preds, masks)
            
            optimizer.step()

            # Update progress bar
            train_bar.set_postfix(loss=running_train_loss / (batch_idx + 1))

        print(f"\t Average Train loss: {running_train_loss/batch_idx:.4f}")
        print(f"\t Train mIoU: {miou_metric.compute().item():.4f}")

        # ####### EVALUATION #######
        int8_miou = qevaluate(prep_model, fp32_model, val_loader)
        if int8_miou > best_val_miou:
            best_val_miou = int8_miou
            save_path = os.path.join(project_dir, "checkpoints", f"{cfg.checkpoint.quant_model_name}.pth")
            torch.save(prep_model.state_dict(), save_path)
            print(f"[QAT] Best model saved to {save_path} with QUANT mIoU = {best_val_miou:.4f}")

        print(f"Epoch {epoch+1} completed")

    print("QAT training completed!")

    # Convert: swap out float ops for quantized kernels
    prep_model = prep_model.to("cpu")
    prep_model.eval()
    quant_model = convert_fx(prep_model)

    # Save the INT8 state_dict
    save_path = os.path.join(project_dir, "checkpoints", f"dino-fpn-qat-int8.pth")
    torch.save(quant_model.state_dict(), save_path)
    print(f"[QuantPTQ] Saved INT8 weights to {save_path}")

    # Print memory footprint
    get_memory_footprint(fp32_model)
    get_quant_memory_footprint(quant_model)


if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="qat"
    ):
        cfg = compose(config_name="qat_config")
        main(cfg)
