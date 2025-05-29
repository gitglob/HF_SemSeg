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

        # pick a few intermediate layers for your laterals (e.g. layers 3, 6, 9)
        idxs = [3, 6, 9]
        laterals = [hidden_states[i][:, 1:] for i in idxs]

        if return_attention:
            return tokens, laterals, backbone_output.attentions
        return tokens, laterals

class SegmentationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, patch_size, lateral_channels=None):
        """
        hidden_size: backbone C
        num_classes: output classes
        patch_size: spatial downsampling factor of backbone
        lateral_channels: list of ints for skip‐connection channels (optional)
        """
        super().__init__()
        self.n_stages = int(math.log2(patch_size))
        # build your upsample blocks & track their out_ch
        self.stages, self.laterals = nn.ModuleList(), nn.ModuleList()
        in_ch = hidden_size
        for i in range(self.n_stages):
            out_ch = in_ch // 2
            # decoder block
            self.stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ))
            # lateral 1×1: map lateral_channels[-1-i] → out_ch
            ch = lateral_channels[-1 - i]
            self.laterals.append(nn.Conv2d(ch, out_ch, kernel_size=1))
            in_ch = out_ch

        self.classifier = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, feat_map, laterals_feats=None):
        """
        feat_map: [B, C, h, w]  from last backbone layer
        laterals_feats: list of [B, C_i, h*2^i, w*2^i] for skip connections
        """
        x = feat_map
        for i, stage in enumerate(self.stages):
            x = stage(x)   # always upsample once per stage

            # only fuse if you have a lateral for *this* stage
            if i < len(laterals_feats):
                lat = laterals_feats[-1 - i]              # pick the matching lateral
                lat = self.laterals[i](lat)               # project to x’s channels
                lat = F.interpolate(
                    lat,
                    size=x.shape[-2:],                    # make it H×W == x
                    mode="bilinear", align_corners=False
                )
                x = x + lat

        logits = self.classifier(x) # [H, W, num_classes]
        return logits
    
class DinoSeg(nn.Module):
    def __init__(self, num_labels=35, model_cfg=None):
        super().__init__()
        BACKBONE_MODEL = model_cfg.backbone

        # Processor
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        self.processor.do_resize      = False
        self.processor.do_center_crop = False
        
        # Backbone
        self.backbone = DinoBackbone(BACKBONE_MODEL, freeze_backbone=model_cfg.freeze_backbone)

        # Segmentation head
        # self.head = SegmentationHead(self.backbone.hidden_size, num_labels, self.backbone.patch_size)
        lateral_channels = [self.backbone.hidden_size]*3 
        self.head = SegmentationHead(
            hidden_size=self.backbone.hidden_size,
            num_classes=num_labels,
            patch_size=self.backbone.patch_size,
            lateral_channels=lateral_channels
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

        # preprocess
        if images.dim() == 3:  # Single image case with shape [C, H, W]
            images = images.unsqueeze(0)  # Add batch dimension to make it [1, C, H, W]
        if images.shape[1] == 1:  # If the image is greyscale
            images = images.repeat(1, 3, 1, 1)  # Convert to 3 channels
        images = self.processor(images, return_tensors="pt").pixel_values.to(device)
        B, _, H, W = images.shape

        # backbone → [B, 1+N, C] and attention maps
        if return_attention:
            tokens, laterals, attentions = tokens # tokens [B, N, C], attentions [B, num_heads, N, N]
        else:
            tokens, laterals = self.backbone(images, return_attention=return_attention)

        # reshape to [B, C, H/patch, W/patch]
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size
        tokens = tokens.transpose(1,2)      # [B, C, N]
        feat = tokens.reshape(B, -1, h, w)  # [B, C, h, w]

        # reshape each lateral the same way
        lateral_feats = []
        for lat in laterals:
            # lat: [B, N, C] → [B, C, h, w]
            lf = lat.transpose(1,2).reshape(B, -1, h, w)
            lateral_feats.append(lf)

        # Upsample and classify
        logits = self.head(feat, laterals_feats=lateral_feats) # [B, num_classes, H, W]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        if return_attention:
            return logits, attentions
        return logits

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = DinoSeg(num_labels=35)

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (3, 375, 1242), device=device))

    print("~~~~~Inference~~~~~")
    images = torch.randn(8, 3, 512, 512) # Example batch of images
    model = model.to(device)
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)
        logits_orig = F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)
        preds = logits_orig.argmax(dim=1)

    print("~~~~~Output Shapes~~~~~")
    print(logits.shape)  # Should be [2, num_labels, H/patch, W/patch]
    print(logits_orig.shape)
    print(preds[1].shape)  # Should be [B, H_out, W_out] with the argmax class per pixel

if __name__ == "__main__":
    main()
