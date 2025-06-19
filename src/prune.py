import sys
import cv2
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import prune
from torchmetrics import JaccardIndex
from tqdm import tqdm
import albumentations as A
import wandb
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

# Add the project root directory to the Python path
cur_dir     = Path(__file__).parent
project_dir = cur_dir.parent
sys.path.append(str(project_dir))

from models.DinoFPNbn import DinoFPN
from models.tools import CombinedLoss
from data.dataset import KittiSemSegDataset
from data.labels_kitti360 import trainId2label, NUM_CLASSES
from utils.visualization import plot_image_and_masks
from utils.others import save_checkpoint, load_checkpoint
from src.train import train_and_validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

"""
Starting from a 30% pruned model with 26% mIoU, we get to 72% after 7 epochs of training.
"""


def apply_global_pruning(model, pruning_ratio=0.3, exclude_layers=True):
    """Global pruning with critical layer protection"""
    
    parameters_to_prune = []
    excluded_layers = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            # Exclude critical layers
            if exclude_layers:
                if any(exclude in name for exclude in ['norm', 'BatchNorm2d', 'layernorm', 'classifier.4']) or \
                isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.Dropout)):
                    excluded_layers.append(name)
                    continue
            
            parameters_to_prune.append((module, 'weight'))
    
    print(f"Pruning {len(parameters_to_prune)} layers")
    # for name, module in parameters_to_prune:
    #     print(f"  - {name}.{module}")
    print(f"Excluded {len(excluded_layers)} critical layers")
    # for layer in excluded_layers:
    #     print(f"  - {layer}")
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    return model

def main(cfg: DictConfig):
    if cfg.wandb.enabled:
        # Initialize wandb
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    crop_size = (cfg.augmentation.crop_height, cfg.augmentation.crop_width)
    train_transform = A.Compose([
        # -- Geometric --
        A.RandomCrop(height=crop_size[0], width=crop_size[1], p=1.0), # preserve scale/context
        A.Affine(
            # translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # ±5% shift
            scale=(0.8, 1.0),                                           # zoom between 0.8×–1.0×
            rotate=(-3, 3),                                             # ±3° roll
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=255,
            fill_mask=255,
            p=0.7
        ),
        A.Perspective(scale=(0.01, 0.03), p=0.5),  # tiny camera viewpoint warp

        # -- Photometric --
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        A.OneOf([
            A.RandomFog(fog_coef_range=(0.05, 0.2), p=1.0),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), p=1.0),
            A.RandomSunFlare(src_radius=50, p=1.0)
        ], p=0.5),

        # -- Occlusions --
        A.CoarseDropout(num_holes_range=(1, 4), 
                        hole_height_range=(5, 30), 
                        hole_width_range=(5, 30), 
                        p=0.5),                                # random occlusion

        # — Blur & noise: motion, sensor, compression —
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=0.4),
            A.GaussianBlur(blur_limit=(3,5), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.2),
        ], p=0.5),
        A.GaussNoise(
            std_range=(10.0/255.0, 50.0/255.0),
            mean_range=(0.0, 0.0),
            p=0.5
        )
    ])

    # Define deterministic transforms for validation
    val_transform = A.Compose([
        A.CenterCrop(height=crop_size[0], width=crop_size[1])
    ])

    # Dataset and DataLoader
    dataset_root = '/home/panos/Documents/data/kitti-360'
    train_dataset = KittiSemSegDataset(dataset_root, train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, 
                              shuffle=True, num_workers=cfg.dataset.num_workers, pin_memory=True)
    val_dataset = KittiSemSegDataset(dataset_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size,
                            shuffle=False, num_workers=cfg.dataset.num_workers, pin_memory=True)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Initialize model, loss function, and optimizer
    model = DinoFPN(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )
    apply_global_pruning(model, pruning_ratio=0.3, exclude_layers=True)
    if not any(p.requires_grad for p in model.backbone.parameters()):
        print("Backbone is frozen.")
    model = model.to(device)
    criterion = CombinedLoss(alpha=0.8, ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Metric: mean IoU over all classes
    miou_metric = JaccardIndex(
        task='multiclass',
        num_classes=cfg.dataset.num_classes,
        average='micro',
        ignore_index=255
    ).to(device)

    # Load the best model if it exists
    start_epoch, best_val_miou = load_checkpoint(model, optimizer, cfg.checkpoint)

    train_and_validate(
        train_loader, val_loader, 
        model, criterion, miou_metric,
        optimizer, cfg, 
        start_epoch=start_epoch,
        best_val_miou=best_val_miou,
        device=device
    )

if __name__ == "__main__":
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="prune_config")
        main(cfg)
