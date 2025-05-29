import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DinoBackbone(nn.Module):
    def __init__(self, backbone_model_name, freeze_backbone=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        self.patch_size = self.backbone.config.patch_size
        self.hidden_size = self.backbone.config.hidden_size

        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, images, return_attention=False):
        backbone_output = self.backbone(
            images, 
            output_attentions=return_attention,
            output_hidden_states=True
        )
        # Extract the hidden states
        hidden_states = backbone_output.hidden_states

        # pick the final tokens for your “main” feature
        tokens = hidden_states[-1][:, 1:] # Exclude the [CLS] token [B, N, C]

        if return_attention:
            return tokens, backbone_output.attentions
        return tokens

class UnetHead(nn.Module):
    def __init__(self, in_channels, n_classes, gn_groups=32):
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.GroupNorm(gn_groups, 512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.GroupNorm(gn_groups, 256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(gn_groups, 128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GroupNorm(gn_groups, 64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=False)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(64, n_classes, 1),
        )

    def forward(self, feat):
        x = self.bottleneck(feat)  # [B, 512, h/32, w/32]
        x = self.up1(x)  # [B, 256, h/16, w/16]
        x = self.up2(x)  # [B, 128, h/8, w/8]
        x = self.up3(x)  # [B, 64, h/8, w/8]
        logits_low = self.classifier(x)  # [B, num_classes, h, w]
        return logits_low

class DinoUnet(nn.Module):
    def __init__(self, num_labels, model_cfg):
        super().__init__()
        BACKBONE_MODEL = model_cfg.backbone

        # Processor
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        # self.processor.do_resize      = False
        # self.processor.do_center_crop = False
        
        # Backbone
        self.backbone = DinoBackbone(BACKBONE_MODEL, freeze_backbone=model_cfg.freeze_backbone)

        # Segmentation head
        self.head = UnetHead(
            in_channels=self.backbone.hidden_size,
            n_classes=num_labels
        )

    def process(self, images):
        # Preprocess the images
        processed_images = self.processor(images, return_tensors="pt").pixel_values
        return processed_images

    def forward(self, images, return_attention=False):
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
        B, _, H, W = images.shape

        # backbone → [B, 1+N, C] and attention maps
        output = self.backbone(images, return_attention=return_attention)
        if return_attention:
            tokens, attentions = output # tokens [B, N, C], attentions [B, num_heads, N, N]
        else:
            tokens = output

        # reshape to [B, C, H/patch, W/patch]
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size
        tokens = tokens.transpose(1,2)      # [B, C, N]
        feat = tokens.reshape(B, -1, h, w)  # [B, C, h, w]

        # Upsample and classify
        logits = self.head(feat) # [B, num_classes, h, w]
        logits = F.interpolate(logits, size=(H_orig, W_orig), mode="bilinear", align_corners=False)

        if return_attention:
            return logits, attentions
        return logits

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = DinoUnet(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    # print(summary(model, (1, 3, H, W), device=device))

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
