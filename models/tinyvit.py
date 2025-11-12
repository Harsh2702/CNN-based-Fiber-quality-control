# models/tinyvit.py
import torch.nn as nn
import timm

class TinyViTBinaryClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'tiny_vit_21m_512.dist_in22k_ft_in1k',
            pretrained=pretrained
        )
        in_features = self.model.head.fc.in_features
        self.model.head.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
