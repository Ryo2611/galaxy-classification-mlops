import torch.nn as nn
from torchvision import models


def build_resnet18(num_outputs: int = 37, pretrained: bool = False, dropout: float = 0.0, **_: object):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features

    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(in_features, num_outputs), nn.Sigmoid()])
    model.fc = nn.Sequential(*layers)
    return model


def build_resnet50(num_outputs: int = 37, pretrained: bool = False, dropout: float = 0.0, **_: object):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features

    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend([nn.Linear(in_features, num_outputs), nn.Sigmoid()])
    model.fc = nn.Sequential(*layers)
    return model
