import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(
    num_outputs: int = 37,
    pretrained: bool = False,
    dropout: float = 0.2,
    **_: object,
):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_outputs),
        nn.Sigmoid(),
    )
    return model
