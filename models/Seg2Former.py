import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchinfo import summary
import torch.nn.functional as F


class Seg2Former(nn.Module):
    def __init__(self, num_labels=None, model_cfg=None):
        """
        A SegFormer‐based semantic segmentation model with a fresh
        classifier head for `num_labels` output classes.
        """
        super().__init__()

        # 1) Load pretrained weights, but override the head size
        #    ignore_mismatched_sizes=True will re‐init any weights that
        #    don’t match the new shape (i.e. the classifier)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        if model_cfg.freeze_backbone:
            for param in self.model.segformer.parameters():
                param.requires_grad = False
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

        # 2) Processor
        self.processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        )

        # 3) build upsample+fuse blocks
        # self.up_block1 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        # )
        # self.up_block2 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        # )
        # self.classifier = nn.Conv2d(64, num_labels, kernel_size=1)

    def process(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self.processor(images, return_tensors="pt").pixel_values  # already floats, normalized
        return pixel_values
    
    def unprocess(self, outputs, images):
        _, _, H, W = images.shape
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=[(H, W)])[0]
        return predicted_semantic_map

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # prepare
        pixel_values = self.process(images).to(self.model.device)
        outputs = self.model(pixel_values=pixel_values)

        logits = outputs.logits                  # Shape: [B, num_labels, h, w]
        # logits_up1 = self.up_block1(logits)      # Upsample to h*2, w*2
        # logits_up2 = self.up_block2(logits_up1)  # Upsample to h*4, w*4
        # logits_up2 = self.classifier(logits_up2)

        logits_up = F.interpolate(logits,
                            size=images.shape[-2:],
                            mode="bilinear",
                            align_corners=False) # Interpolate to H, W

        return logits_up


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = Seg2Former(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    ).to(device)

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (1, 3, H, W), device=device))

    print("~~~~~Inference~~~~~")
    model.eval()
    print("~~~~~Input Shapes~~~~~")
    images = torch.randint(0, 256, (1, 3, H, W), dtype=torch.uint8)
    print(f"Input  shape: {tuple(images.shape)}")
    with torch.no_grad():
        # Forward
        logits = model(images)
        preds = logits.argmax(dim=1)

    print("~~~~~Output Shapes~~~~~")
    print(logits.shape)  # Should be [2, num_labels, H/patch, W/patch]
    print(preds.shape)  # Should be [B, H_out, W_out] with the argmax class per pixel

if __name__ == "__main__":
    from hydra import compose, initialize
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
