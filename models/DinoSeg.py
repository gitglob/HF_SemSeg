import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor


class DinoSeg(nn.Module):
    def __init__(self, num_labels=35):
        super().__init__()
        BACKBONE_MODEL = "facebook/dinov2-small"

        # 1) backbone and its processor
        self.backbone  = AutoModel.from_pretrained(BACKBONE_MODEL)
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        # 2) head: 1×1 conv on the ViT feature map
        hidden_size    = self.backbone.config.hidden_size
        self.classifier = nn.Conv2d(hidden_size, num_labels, kernel_size=1)
        # patch grid size = image_size/patch_size, will infer in forward
        self.patch_size = self.backbone.config.patch_size

    def process(self, images):
        # Preprocess the images
        processed_images = self.processor(images, return_tensors="pt").pixel_values
        return processed_images

    def forward(self, images, original_size=None, return_pred=False):
        # preprocess
        if not torch.is_tensor(images):
            pixel_vals = self.processor(images, return_tensors="pt").pixel_values
        else:
            if images.dim() == 3:  # Single image case with shape [C, H, W]
                images = images.unsqueeze(0)  # Add batch dimension to make it [1, C, H, W]
            
            if images.shape[1] == 1:  # If the image is greyscale
                pixel_vals = images.repeat(1, 3, 1, 1)  # Convert to 3 channels
            else:
                pixel_vals = images  # Assume [B, 3, H, W] already

        B, _, H, W = pixel_vals.shape
        # backbone → [B, 1+N, C]
        tokens = self.backbone(pixel_vals).last_hidden_state[:, 1:]
        # reshape to [B, C, H/patch, W/patch]
        h, w = H // self.patch_size, W // self.patch_size
        feat = tokens.transpose(1,2).reshape(B, -1, h, w)
        # classify and optional upsample
        logits = self.classifier(feat)
        if original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        
        # 6) optionally return per-pixel preds
        if return_pred:
            # preds: [B, H_out, W_out] with the argmax class per pixel
            preds = logits.argmax(dim=1)
            return logits, preds

        return logits

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    # Example usage
    model = DinoSeg(num_labels=35).to(device)
    images = torch.randn(8, 3, 512, 512).to(device)  # Example batch of images
    with torch.no_grad():
        outputs = model(images)
        outputs_orig = model(images, original_size=orig_size)
        preds = model(images, original_size=orig_size, return_pred=True)
    print(outputs.shape)  # Should be [2, num_labels, H/patch, W/patch]
    print(outputs_orig.shape)
    print(preds[1].shape)  # Should be [B, H_out, W_out] with the argmax class per pixel

if __name__ == "__main__":
    main()