import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FPNHead(nn.Module):
    def __init__(self, 
                 embed_dims: list[int],   # list of C at each layer you tap
                 proj_channels: int,
                 num_classes: int,
                 patch_size: int):
        """
        embed_dims: e.g. [384, 384, 384, 384] if each hidden-state has 384 dims
        proj_channels: e.g. 256
        """
        super().__init__()
        self.patch_size = patch_size

        # 1×1 proj for each feature
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, proj_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, proj_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )
                for in_ch in embed_dims
        ])
        
        # final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(len(embed_dims)*proj_channels, proj_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, proj_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(proj_channels, num_classes, kernel_size=1),
        )

    def forward(self, hidden_states, orig_size):
        """
        hidden_states: tuple of [B, N, D] from several layers
        returns: logits [B, num_classes, H, W]
        """
        H_orig, W_orig = orig_size

        B = hidden_states[0].size(0)
        feats = []

        # Forward each hidden state
        # turn each [B, N, D] → [B, D, Hf, Wf]
        for hs, proj in zip(hidden_states, self.projs):
            # skip the cls token
            hs = hs[:, 1:]                      # → [B, N-1, D]

            Hf = H_orig // self.patch_size
            Wf = W_orig // self.patch_size

            feat2d = hs.transpose(1, 2)         # → [B, D, N-1]
            feat2d = feat2d.view(B, -1, Hf, Wf) # → [B, D, Hf, Wf]
            feats.append(proj(feat2d))          # → [B, proj_channels, Hf, Wf]

        # predict low-res logits and then upsample to orig
        fused = torch.cat(feats, dim=1)       # [B, len(embed_dims)*proj_channels, Hf, Wf]
        logits_low = self.classifier(fused)   # [B, num_classes, Hf, Wf]
        return logits_low

class DinoFPN(nn.Module):
    def __init__(self, num_labels, model_cfg):
        super().__init__()
        BACKBONE_MODEL = model_cfg.backbone

        # Processor
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        self.processor.do_resize = False
        self.processor.do_center_crop = False

        # Backbone
        self.backbone = AutoModel.from_pretrained(BACKBONE_MODEL)
        self.patch_size = self.backbone.config.patch_size
        self.hidden_size = self.backbone.config.hidden_size
        # optionally freeze
        if model_cfg.freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
            
        # Segmentation head
        self.head = FPNHead(
            embed_dims=[self.backbone.config.hidden_size] * 4,
            proj_channels=256,
            num_classes=num_labels,
            patch_size=self.backbone.config.patch_size
        )
        self.layer_idxs = model_cfg.deepsup_layers

    def process(self, images):
        # Preprocess the images
        processed_images = self.processor(images, return_tensors="pt").pixel_values
        return processed_images

    def forward(self, images):
        """
        B: batch size
        N: number of feature tokens
        C: hidden dimension
        H: height of the image
        W: width of the image
        """
        B, _, H_orig, W_orig = images.shape

        # preprocess
        if images.dim() == 3:  # Single image case with shape [C, H, W]
            images = images.unsqueeze(0)  # Add batch dimension to make it [1, C, H, W]
        if images.shape[1] == 1:  # If the image is greyscale
            images = images.repeat(1, 3, 1, 1)  # Convert to 3 channels

        images = self.processor(images, return_tensors="pt").pixel_values.to(device)

        # backbone → [B, 1+N, C] and attention maps
        out = self.backbone(images, output_hidden_states=True)
        hstates = out.hidden_states # [13] [B, 1+N, D]
        taps = [hstates[i] for i in self.layer_idxs]

        # Upsample and classify
        logits_low = self.head(taps, (H_orig, W_orig)) # [B, num_classes, h, w]
        logits = F.interpolate(
            logits_low,
            size=(H_orig, W_orig),
            mode="bilinear",
            align_corners=False
        )

        return logits

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = DinoFPN(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (2, 3, H, W), device=device))

    print("~~~~~Inference~~~~~")
    images = torch.randint(0, 256, (8, 3, H, W), dtype=torch.uint8)  # Example batch of images in 0~255
    model = model.to(device)
    images = images.to(device)
    with torch.no_grad():
        images_processed = model.process(images)
        print(f"Input  shape: {tuple(images.shape)}, type: {images.dtype}, (min, max): ({images.min()}, {images.max()})")
        print(f"Processed shape: {tuple(images_processed.shape)}, type: {images_processed.dtype}, (min, max): ({images_processed.min()}, {images_processed.max()})")

        logits = model(images)
        preds = logits.argmax(dim=1)

    print("~~~~~Output Shapes~~~~~")
    print(logits.shape)  # Should be [2, num_labels, H/patch, W/patch]
    print(preds[1].shape)  # Should be [B, H_out, W_out] with the argmax class per pixel

if __name__ == "__main__":
    from hydra import compose, initialize
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
