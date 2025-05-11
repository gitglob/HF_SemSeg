import torch.nn as nn
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig, AutoModel, AutoImageProcessor


class DinoSeg2Former(nn.Module):
    def __init__(self, num_labels=35):
        super(DinoSeg2Former, self).__init__()

        BACKBONE_MODEL = 'facebook/dinov2-small'

        backbone_model = AutoModel.from_pretrained(BACKBONE_MODEL)

        BASE_MODEL = "facebook/mask2former-swin-small-ade-semantic"

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(BASE_MODEL)
        cfg = Mask2FormerConfig.from_pretrained(BASE_MODEL)
        cfg.d_model           = backbone_model.config.hidden_size
        cfg.hidden_size       = backbone_model.config.hidden_size
        cfg.num_labels        = num_labels

        # Segformer head
        self.model = Mask2FormerForUniversalSegmentation(cfg)
        self.model.model.encoder = backbone_model
        # re-init the query embeddings
        self.model.model.query_embed = nn.Embedding(cfg.num_queries, cfg.d_model)
        # re-init the linear layer that turns each query into a class score
        self.model.model.class_queries_proj = nn.Linear(cfg.d_model, cfg.num_labels)

        # Set the model pre-processor
        backbone_processor = AutoImageProcessor.from_pretrained(BACKBONE_MODEL, use_fast=False)
        self.processor = backbone_processor

    def process(self, images):
        # Preprocess the images
        processed_images = self.processor(images, return_tensors="pt").pixel_values
        return processed_images

    def forward(self, images):
        if not isinstance(images, torch.Tensor):
            images = self.processor(images)["pixel_values"]
        outputs = self.model(images)
        return outputs

    def load_pretrained_weights(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))