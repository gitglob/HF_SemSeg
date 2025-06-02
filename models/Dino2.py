import torch
import torch.nn as nn
from torch.nn import functional as F
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dino(nn.Module):
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
        if model_cfg.freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
            
        # Segmentation head
        self.classifier = nn.Sequential(
            nn.Conv2d(self.hidden_size, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Upsample(scale_factor=self.patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_labels, kernel_size=1),
        )

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
        out = self.backbone(images, output_hidden_states=False)
        feats = out.last_hidden_state # [B, 1+N, D]
        feats = feats[:, 1:, :]  # Remove the first token (CLS token), now [B, N, D]
        Hf = H_orig // self.patch_size
        Wf = W_orig // self.patch_size
        feats2d = feats.permute(0, 2, 1).reshape(B, -1, Hf, Wf)  # [B, D, Hf, Wf]

        # Upsample and classify
        logits_low = self.classifier(feats2d) # [B, num_classes, h, w]
        logits = F.interpolate(              # [B, num_classes, H_orig, W_orig]
            logits_low,
            size=(H_orig, W_orig),
            mode="bilinear",
            align_corners=False
        ) # Final interpolation, in case the logits from the head are not at the original resolution

        return logits

    def load_pretrained_weights(self, weight_path):
        self.load_state_dict(torch.load(weight_path, map_location='cpu'))


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 280
    W = 840
    orig_size = (H, W)
    model = Dino(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (2, 3, H, W), device=device))

    print("~~~~~Inference~~~~~")
    images = torch.randint(0, 256, (2, 3, H, W), dtype=torch.uint8)  # Example batch of images in 0~255
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
