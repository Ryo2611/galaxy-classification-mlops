import os
import sys
from dataclasses import dataclass

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.model_factory import build_model_from_config  # noqa: E402


TARGET_COLS = [
    "Class1.1", "Class1.2", "Class1.3", "Class2.1", "Class2.2",
    "Class3.1", "Class3.2", "Class4.1", "Class4.2",
    "Class5.1", "Class5.2", "Class5.3", "Class5.4",
    "Class6.1", "Class6.2", "Class7.1", "Class7.2", "Class7.3",
    "Class8.1", "Class8.2", "Class8.3", "Class8.4",
    "Class8.5", "Class8.6", "Class8.7",
    "Class9.1", "Class9.2", "Class9.3",
    "Class10.1", "Class10.2", "Class10.3",
    "Class11.1", "Class11.2", "Class11.3", "Class11.4", "Class11.5", "Class11.6",
]


@dataclass
class ModelBundle:
    model: torch.nn.Module
    device: torch.device
    model_name: str
    checkpoint_path: str
    checkpoint_loaded: bool
    target_cols: list[str]
    supports_gradcam: bool


MODEL_REGISTRY = {
    "resnet50": {
        "label": "ResNet50 baseline",
        "config": os.path.join(PROJECT_ROOT, "configs", "resnet50.yaml"),
        "checkpoint": os.path.join(PROJECT_ROOT, "models", "baseline_resnet50_best.pth"),
        "supports_gradcam": True,
    },
    "resnet18": {
        "label": "ResNet18 baseline",
        "config": os.path.join(PROJECT_ROOT, "configs", "resnet18.yaml"),
        "checkpoint": os.path.join(PROJECT_ROOT, "models", "resnet18_baseline_best.pth"),
        "supports_gradcam": True,
    },
    "efficientnet_b0": {
        "label": "EfficientNet-B0 comparison",
        "config": os.path.join(PROJECT_ROOT, "configs", "efficientnet_b0.yaml"),
        "checkpoint": os.path.join(PROJECT_ROOT, "models", "efficientnet_b0_comparison_best.pth"),
        "supports_gradcam": True,
    },
    "convnext_tiny": {
        "label": "ConvNeXt-Tiny comparison",
        "config": os.path.join(PROJECT_ROOT, "configs", "convnext_tiny.yaml"),
        "checkpoint": os.path.join(PROJECT_ROOT, "models", "convnext_tiny_comparison_best.pth"),
        "supports_gradcam": True,
    },
    "vit_b_16": {
        "label": "ViT-B/16 comparison",
        "config": os.path.join(PROJECT_ROOT, "configs", "vit_b_16.yaml"),
        "checkpoint": os.path.join(PROJECT_ROOT, "models", "vit_b_16_comparison_best.pth"),
        "supports_gradcam": False,
    },
}

_MODEL_CACHE: dict[str, ModelBundle] = {}


def resolve_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def available_models() -> list[dict]:
    models = []
    for name, spec in MODEL_REGISTRY.items():
        checkpoint_path = spec["checkpoint"]
        models.append({
            "name": name,
            "label": spec["label"],
            "checkpoint_path": checkpoint_path,
            "checkpoint_loaded": os.path.exists(checkpoint_path),
            "supports_gradcam": spec["supports_gradcam"],
        })
    return models


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_name: str | None = None) -> ModelBundle:
    model_name = model_name or os.getenv("MODEL_NAME", "resnet50")
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    spec = MODEL_REGISTRY[model_name]
    checkpoint_path = os.getenv("MODEL_PATH", spec["checkpoint"]) if model_name == os.getenv("MODEL_NAME") else spec["checkpoint"]
    device = resolve_device()

    config = load_config(spec["config"])
    model = build_model_from_config(config, num_outputs=len(TARGET_COLS))
    checkpoint_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            checkpoint_loaded = True
        except RuntimeError as exc:
            print(f"Warning: failed to load checkpoint '{checkpoint_path}': {exc}", file=sys.stderr)
    else:
        print(f"Warning: checkpoint not found at '{checkpoint_path}'. Using initialized weights.", file=sys.stderr)

    model.to(device)
    model.eval()
    bundle = ModelBundle(
        model=model,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_loaded=checkpoint_loaded,
        target_cols=TARGET_COLS,
        supports_gradcam=spec["supports_gradcam"],
    )
    _MODEL_CACHE[model_name] = bundle
    return bundle
