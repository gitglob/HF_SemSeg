import torch
import torch.nn as nn
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from torchinfo import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seg2Former(nn.Module):
    def __init__(self, num_labels=19, model_cfg=None):
        """
        A SegFormer‐based semantic segmentation model with a fresh
        classifier head for `num_labels` output classes.
        """
        super().__init__()

        # 1) Load pretrained weights, but override the head size
        #    ignore_mismatched_sizes=True will re‐init any weights that
        #    don’t match the new shape (i.e. the classifier)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        # 1) Freeze the entire encoder
        for param in self.model.segformer.parameters():
            param.requires_grad = False

        # 2) (Re-)initialize the decoder head and make sure it's trainable
        #    If you've loaded with ignore_mismatched_sizes=True, HuggingFace
        #    already reinitialized decode_head for num_labels!=21.
        for param in self.model.decode_head.parameters():
            param.requires_grad = True

        # 2) Update processor so it won't resize/crop for you
        self.processor = AutoImageProcessor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            use_fast=False
        )

    def process(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self.processor(images, return_tensors="pt").pixel_values  # already floats, normalized
        return pixel_values

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # prepare
        pixel_values = self.process(images).to(self.model.device)

        # forward through SegFormer
        outputs = self.model(pixel_values=pixel_values)

        # outputs.logits is [B, num_labels, H, W]
        return outputs.logits


if __name__ == "__main__":
    model = Seg2Former(num_labels=35).to(device)

    print("~~~~~Processor~~~~~")
    print(model.processor)

    print("~~~~~Model~~~~~")
    print(summary(model, (1, 3, 512, 512), device=device))

    print("~~~~~Inference~~~~~")
    model.eval()
    print("~~~~~Input Shapes~~~~~")
    dummy_imgs = torch.randn(2, 3, 512, 512).to(device)
    print(f"Input  shape: {tuple(dummy_imgs.shape)}")
    with torch.no_grad():
        # Forward
        logits = model(dummy_imgs)
        preds = logits.argmax(dim=1)

    print("~~~~~Output Shapes~~~~~")
    print(logits.shape)  # Should be [2, num_labels, H/patch, W/patch]
    print(preds[1].shape)  # Should be [B, H_out, W_out] with the argmax class per pixel
