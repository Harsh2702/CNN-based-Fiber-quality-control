# models/swinv2_tiny.py
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

class SwinV2TinyClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        model_name = "microsoft/swinv2-tiny-patch4-window16-256"

        # Load pretrained Hugging Face model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.base_model = AutoModelForImageClassification.from_pretrained(model_name)

        # Replace classification head to match num_classes
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, images):
        """
        images: batch of images (PIL or tensors)
        returns: logits
        """
        # Preprocess input
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.base_model(**inputs)
        return outputs.logits