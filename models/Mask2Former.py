import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from torchinfo import summary
import torch.nn.functional as F


class Mask2Former(nn.Module):
    def __init__(self, num_labels=None, model_cfg=None):
        """
        A Mask2Former semantic segmentation model with a fresh
        classifier head for `num_labels` output classes.
        """
        super().__init__()

        # 1) Load pretrained weights, but override the head size
        #    ignore_mismatched_sizes=True will re‐init any weights that
        #    don’t match the new shape (i.e. the classifier)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-semantic",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        if model_cfg.freeze_backbone:
            for param in self.model.model.parameters():
                param.requires_grad = False
        for param in self.model.class_predictor.parameters():
            param.requires_grad = True

        # 2) Processor
        self.processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-semantic",
            reduce_labels=True,
        )

    def process(self, images_np: np.ndarray, gt_masks_np: np.ndarray):
        images_np = [images_np[i] for i in range(images_np.shape[0])]
        gt_masks_np = [gt_masks_np[i] for i in range(gt_masks_np.shape[0])]
        inputs = self.processor(
            images_np,                    # list or tensor of images
            gt_masks_np,                  # list or tensor of segmentation maps
            return_tensors="pt"
        )
        return inputs

    def unprocess(self, outputs, images):
        B, _, H, W = images.shape
        target_sizes = [(H, W)] * B
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        predicted_semantic_map = torch.stack(predicted_semantic_map, dim=0)  # [B, H, W]
        return predicted_semantic_map

    def forward(self, images: np.ndarray, masks: np.ndarray) -> torch.Tensor:
        # prepare
        inputs = self.process(images, masks)

        # forward
        pixel_values = inputs.pixel_values.to(self.model.device)
        mask_labels = [mask_label.to(self.model.device) for mask_label in inputs.mask_labels]
        class_labels = [class_label.to(self.model.device) for class_label in inputs.class_labels]
        outputs = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        # print(f"Class logits shape: {outputs.class_queries_logits.shape}")
        # print(f"Mask logits shape: {outputs.masks_queries_logits.shape}")
        loss = outputs.loss
        preds = self.unprocess(outputs, images)

        return preds, loss


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    H = 375
    W = 1242
    orig_size = (H, W)
    model = Mask2Former(
        num_labels=cfg.dataset.num_classes,
        model_cfg=cfg.model
    ).to(device)

    print("~~~~~Processor~~~~~")
    # print(model.processor)

    print("~~~~~Model~~~~~")
    # print(summary(model, (1, 3, H, W), device=device))

    print("~~~~~Inference~~~~~")
    model.eval()
    print("~~~~~Input Shapes~~~~~")
    images = np.random.randint(0, 256, (4, 3, H, W), dtype=np.uint8)
    gt_masks = np.random.randint(0, cfg.dataset.num_classes, (4, H, W), dtype=np.int64)
    print(f"Input  shape: {tuple(images.shape)}")
    print(f"GT Mask shape: {tuple(gt_masks.shape)}")
    with torch.no_grad():
        # Forward
        preds, loss = model(images, gt_masks)

    print("~~~~~Output Shapes~~~~~")
    print(f"Loss: {loss.item()}")
    print("Preds: ", preds.shape)  # Should be [B, H_out, W_out] with the argmax class per pixel

if __name__ == "__main__":
    from hydra import compose, initialize
    with initialize(
        version_base=None, 
        config_path=f"../configs", 
        job_name="train_and_log"
    ):
        cfg = compose(config_name="config")
        main(cfg)
