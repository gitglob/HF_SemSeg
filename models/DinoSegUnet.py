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

        # pick a few intermediate layers the laterals (e.g. layers 3, 6, 9)
        idxs = [3, 6, 9]
        laterals = [hidden_states[i][:, 1:] for i in idxs] # each lateral: [B, 256, 768]

        if return_attention:
            return tokens, laterals, backbone_output.attentions
        return tokens, laterals

class EncoderDecoderHead(nn.Module):
    def __init__(self, in_channels, skip_channels, decoder_channels, n_classes):
        super().__init__()
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels), 
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        # Projections for each skip
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, decoder_channels,1), 
                          nn.BatchNorm2d(decoder_channels), 
                          nn.ReLU(),
                          nn.Dropout2d(0.1)
                          )
            for c in skip_channels
        ])
        # Refinement convs after each fuse
        self.refines = nn.ModuleList([
            nn.Sequential(nn.Conv2d(decoder_channels, decoder_channels,3,padding=1),
                          nn.BatchNorm2d(decoder_channels), 
                          nn.ReLU(),
                          nn.Dropout2d(0.1),
                          nn.Conv2d(decoder_channels, decoder_channels,3,padding=1),
                          nn.BatchNorm2d(decoder_channels), 
                          nn.ReLU(),
                          nn.Dropout2d(0.1)
            )
            for _ in skip_channels
        ])
        # Final classifier
        self.dropout = nn.Dropout2d(0.1)
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, n_classes, 1),
        )

    def forward(self, feat, skips):
        """
        Take the output from the backbone and the lateral features:
            1. Pass the output through the bottleneck
            2. For each skip:
                a. Project the skip to the decoder channels
                b. Add the projected skip to the output
                c. Pass through a refinement conv to combine different scale features
            3. Classify the output to get the final logits
        """
        # feat: deepest [B,C4,h/32,w/32]
        # skips: list of skip maps [feat3, feat2, feat1]
        x = self.bottleneck(feat)
        for proj, refine, skip in zip(self.proj, self.refines, skips):
            # x:    [B, N, h, w] # N < C
            # skip: [B, C, h, w]
            skip = proj(skip) # [B, N, h, w]
            x = x + skip # [B, N, h, w]
            x = refine(x) # [B, N, h, w]
        logits = self.classifier(x) # [B, num_classes, h, w]
        return logits
    
class DinoSegUnet(nn.Module):
    def __init__(self, num_labels, model_cfg):
        super().__init__()
        BACKBONE_MODEL = model_cfg.backbone

        # Processor
        self.processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        self.processor.do_resize      = False
        self.processor.do_center_crop = False
        
        # Backbone
        self.backbone = DinoBackbone(BACKBONE_MODEL, freeze_backbone=model_cfg.freeze_backbone)

        # Segmentation head
        lateral_channels = [self.backbone.hidden_size]*3 
        self.head = EncoderDecoderHead(
            in_channels=self.backbone.hidden_size,
            skip_channels=lateral_channels,
            decoder_channels=model_cfg.decoder_channels,
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
            tokens, laterals, attentions = output # tokens [B, N, C], attentions [B, num_heads, N, N]
        else:
            tokens, laterals = output

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
        logits = self.head(feat, skips=lateral_feats) # [B, num_classes, h, w]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

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
    model = DinoSegUnet(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    )

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (3, 375, 1242), device=device))

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
