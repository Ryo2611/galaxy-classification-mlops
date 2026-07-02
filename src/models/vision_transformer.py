import torch.nn as nn
from torchvision import models


def build_vit_b_16(
    num_outputs: int = 37,
    pretrained: bool = False,
    dropout: float = 0.0,
    image_size: int = 224,
    **_: object,
):
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights, image_size=image_size)
    in_features = model.heads.head.in_features

    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(in_features, num_outputs), nn.Sigmoid()])
    model.heads.head = nn.Sequential(*layers)
    return model
