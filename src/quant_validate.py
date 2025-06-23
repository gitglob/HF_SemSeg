import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm
import albumentations as A
from hydra import initialize, compose
from omegaconf import DictConfig

# Quantization imports
from torch.ao.quantization import QConfig
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from transformers.models.dinov2.modeling_dinov2 import Dinov2PatchEmbeddings, Dinov2Embeddings

# Add the project root directory to the Python path
cur_file    = Path(__file__)
cur_dir     = cur_file.parent
project_dir = cur_dir.parent
sys.path.append(str(project_dir))

from models.DinoFPNbn import DinoFPN as DinoSeg
from models.tools import CombinedLoss
from data.kitti360.dataset import KittiSemSegDataset
from utils.others import get_quant_memory_footprint


def create_quantized_model(cfg, fp32_model):
    """Create and load a quantized model"""
    # Set up quantization config
    activation = HistogramObserver.with_args(
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine
    )
    weight = PerChannelMinMaxObserver.with_args(
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0
    )
    custom_qconfig = QConfig(activation=activation, weight=weight)
    
    qconfig_map = (
        QConfigMapping()
        .set_global(custom_qconfig)
        .set_module_name("backbone.embeddings.patch_embeddings", None)
        .set_module_name("backbone.embeddings.dropout", None)
        .set_object_type("size", None)
        .set_object_type("view", None)
        .set_object_type("reshape", None)
        .set_object_type("permute", None)
        .set_object_type(torch.Tensor.size, None)
        .set_object_type(torch.Tensor.view, None)
        .set_object_type(torch.Tensor.reshape, None)
    )
    
    prepare_custom_config = (
        PrepareCustomConfig()
        .set_non_traceable_module_classes([
            Dinov2PatchEmbeddings, 
            Dinov2Embeddings
        ])
    )
    
    # Prepare the model for quantization
    example_inputs = (torch.randn(1, 3, cfg.augmentation.crop_height, cfg.augmentation.crop_width),)
    prep_model = prepare_fx(
        fp32_model, 
        qconfig_map, 
        example_inputs,
        prepare_custom_config=prepare_custom_config
    )
    
    # Convert to quantized model
    quant_model = convert_fx(prep_model)
    
    return quant_model


def main(cfg: DictConfig, use_quantized=True):
    device = torch.device('cuda' if torch.cuda.is_available() and not use_quantized else 'cpu')
    print(f"Using device: {device}")

    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    val_transform = A.Compose([
        A.CenterCrop(crop_size[0], crop_size[1])
    ])

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    print(f"Validation dataset size: {len(val_dataset)}")

    # Load the fp32 model
    fp32_model = DinoSeg(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )
    fp32_model.eval()

    # Set up the checkpoint path
    checkpoint_filename = cfg.checkpoint.quant_model_name + ".pth"
    checkpoint_path = project_dir / "checkpoints" / checkpoint_filename

    # Load quantized model
    model_type = "INT8 Quantized"
    model = create_quantized_model(cfg, fp32_model)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    # Print memory footprint
    get_quant_memory_footprint(model)

    model = model.to(device)
    model.eval()
    criterion = CombinedLoss(alpha=0.8, ignore_index=255)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='micro',
        ignore_index=255
    )

    ####### VALIDATION #######
    running_val_loss = 0.0
    miou_metric.reset()

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(val_loader, desc=f"Validating {model_type}")):
            imgs = imgs.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
            # Use fp32_model for preprocessing (same as in your notebook)
            imgs = fp32_model.process(imgs).to(device)

            # Forward pass through model (quantized or fp32)
            logits = model(imgs)

            # Loss (move to same device as logits)
            masks = masks.to(logits.device)
            loss = criterion(logits, masks.long())

            # accumulate losses
            running_val_loss += loss.item()

            # compute IoU on this batch
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
            miou_metric.update(preds.cpu(), masks.cpu())

        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_miou = miou_metric.compute().item()

        ####### PRINT RESULTS #######
        print(f"\n{model_type} Model Results:")
        print(f"  Val Loss: {avg_val_loss:.4f} | mIoU: {avg_val_miou:.4f}")


if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="validate_quantized"
    ):
        # Run validation on quantized model
        cfg = compose(config_name="ptq_config")
        print("=== Validating Quantized Model ===")
        main(cfg)
