import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, patch_size):
        super().__init__()
        self.proj    = nn.Conv2d(hidden_size, hidden_size//2, kernel_size=1)
        self.up      = nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False)
        self.classifier = nn.Conv2d(hidden_size//2, num_classes, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He normalization for conv layers feeding into ReLUs
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # If you ever use BN, weight=1, bias=0 is standard
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, feat_map):       # (B, hidden, h, w) , hxw is all the backbone feature tokens
        x = self.proj(feat_map)        # (B, hidden//2, h, w) - project to 1/2 hidden size
        x = self.up(x)                 # (B, hidden//2, H, W), upsample to original image size
        logits = self.classifier(x)    # (B, num_classes, H, W) - classify each pixel
        return logits

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
        backbone_output = self.backbone(images, output_attentions=return_attention)
        tokens = backbone_output.last_hidden_state[:, 1:]  # Exclude the [CLS] token [B, N, C]

        if return_attention:
            return tokens, backbone_output.attentions
        return tokens
    
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
        self.head = SegmentationHead(self.backbone.hidden_size, num_labels, self.backbone.patch_size)

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
        tokens = self.backbone(images, return_attention=return_attention)
        if return_attention:
            tokens, attentions = tokens # tokens [B, N, C], attentions [B, num_heads, N, N]

        # reshape to [B, C, H/patch, W/patch]
        h, w = H // self.backbone.patch_size, W // self.backbone.patch_size
        tokens = tokens.transpose(1,2)      # [B, C, N]
        feat = tokens.reshape(B, -1, h, w)  # [B, C, h, w]

        # Upsample and classify
        logits = self.head(feat)           # [B, num_classes, H, W]
    
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
    print(summary(model, (3, 375, 1242), device="cpu"))

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
