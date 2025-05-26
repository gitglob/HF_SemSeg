import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torchinfo import summary
from transformers import AutoModel, AutoImageProcessor


class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, skip_channels, decoder_channels, n_classes, gn_groups=32):
        super().__init__()
        assert decoder_channels % gn_groups == 0, "decoder_channels must be divisible by gn_groups"
        assert gn_groups >= 1, "gn_groups must be at least 1"

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(gn_groups, decoder_channels),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        # Projections for skip connections
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_channels, kernel_size=1),
                nn.GroupNorm(gn_groups, decoder_channels),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            )
            for c in skip_channels
        ])
        # Refinement convs after each fuse
        self.refines = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
                nn.GroupNorm(gn_groups, decoder_channels),
                nn.ReLU(),
                nn.Dropout2d(0.1),
                nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
                nn.GroupNorm(gn_groups, decoder_channels),
                nn.ReLU(),
                nn.Dropout2d(0.1)
            )
            for _ in skip_channels
        ])
        # Final classifier for main output
        self.classifier = nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
        # Auxiliary classifiers for deep supervision at each skip scale
        self.aux_classifiers = nn.ModuleList([
            nn.Conv2d(decoder_channels, n_classes, kernel_size=1)
            for _ in skip_channels
        ])

    def forward(self, feat, skips):
        # feat: [B, C, h, w]
        # skips: list of [B, C, h, w] intermediate features
        x = self.bottleneck(feat)
        aux_outputs = []
        for i, (proj, refine, skip) in enumerate(zip(self.proj, self.refines, skips)):
            skip_proj = proj(skip)
            x = x + skip_proj
            x = refine(x)
            # produce an auxiliary segmentation map at this scale
            aux_logits = self.aux_classifiers[i](x)
            aux_outputs.append(aux_logits)
        # main output at final scale
        main_logits = self.classifier(x)
        return main_logits, aux_outputs


class DinoSegDeepSup(nn.Module):
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

        # Decoder head with deep supervision
        skip_ch = [self.hidden_size] * len(model_cfg.deepsup_layers)
        self.head = DeepSupervisionHead(
            in_channels=self.hidden_size,
            skip_channels=skip_ch,
            decoder_channels=model_cfg.decoder_channels,
            n_classes=num_labels
        )
        self.deepsup_layers = model_cfg.deepsup_layers  # e.g. [3,6,9]

    def process(self, images):
        # Preprocess the images
        processed_images = self.processor(images, return_tensors="pt").pixel_values
        return processed_images

    def forward(self, images):
        # preprocess
        if images.dim() == 3: images = images.unsqueeze(0)
        if images.shape[1] == 1: images = images.repeat(1,3,1,1)
        pix = self.processor(images, return_tensors="pt").pixel_values.to(images.device)
        B, _, H, W = pix.shape

        # backbone
        out = self.backbone(pix, output_hidden_states=True)
        hstates = out.hidden_states
        # skip features and main tokens
        laterals = [hstates[i][:,1:].transpose(1,2).reshape(B, -1, H//self.patch_size, W//self.patch_size)
                        for i in self.deepsup_layers]
        tokens = hstates[-1][:,1:]
        feat = tokens.transpose(1,2).reshape(B, -1, H//self.patch_size, W//self.patch_size)

        # decoder
        main_logits, aux_logits = self.head(feat, laterals)
        # upsample all to input resolution
        main_up = F.interpolate(main_logits, size=(H,W), mode="bilinear", align_corners=False)
        aux_ups = [F.interpolate(a, size=(H,W), mode="bilinear", align_corners=False)
                    for a in aux_logits]
        return main_up, aux_ups


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = DinoSegDeepSup(
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

        logits, aux_logits = model(images)
        preds = logits.argmax(dim=1)
        aux_preds = [a.argmax(dim=1) for a in aux_logits]

    print("~~~~~Output Shapes~~~~~")
    for i, aux_logits in enumerate(aux_logits):
        print(f"Auxiliary output {i} shape: {aux_logits.shape}")
    print(logits.shape)  # Should be [2, num_labels, H/patch, W/patch]
    for i, aux in enumerate(aux_preds):
        print(f"Auxiliary output {i} shape: {aux.shape}")
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
