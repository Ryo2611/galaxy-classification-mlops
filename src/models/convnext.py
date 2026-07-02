import torch.nn as nn
from torchvision import models


def build_convnext_tiny(
    num_outputs: int = 37,
    pretrained: bool = False,
    dropout: float = 0.2,
    **_: object,
):
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_outputs),
        nn.Sigmoid(),
    )
    return model
