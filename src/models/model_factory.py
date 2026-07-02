from src.models.convnext import build_convnext_tiny
from src.models.efficientnet import build_efficientnet_b0
from src.models.resnet import build_resnet18, build_resnet50
from src.models.vision_transformer import build_vit_b_16


SUPPORTED_MODELS = {
    "resnet18": build_resnet18,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "convnext_tiny": build_convnext_tiny,
    "vit_b_16": build_vit_b_16,
}


def build_model_from_name(
    name: str,
    num_outputs: int = 37,
    pretrained: bool = False,
    dropout: float = 0.0,
    image_size: int = 224,
):
    if name not in SUPPORTED_MODELS:
        supported = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}")
    return SUPPORTED_MODELS[name](
        num_outputs=num_outputs,
        pretrained=pretrained,
        dropout=dropout,
        image_size=image_size,
    )


def build_model_from_config(config: dict, num_outputs: int = 37):
    model_cfg = config["model"]
    name = model_cfg.get("name", model_cfg.get("architecture", "resnet18"))
    return build_model_from_name(
        name=name,
        num_outputs=num_outputs,
        pretrained=model_cfg.get("pretrained", False),
        dropout=model_cfg.get("dropout", 0.0),
        image_size=config.get("preprocessing", {}).get("image_size", 224),
    )
