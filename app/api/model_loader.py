import os
import sys
from dataclasses import dataclass

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.resnet import build_resnet50  # noqa: E402


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


def resolve_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def load_model() -> ModelBundle:
    model_name = os.getenv("MODEL_NAME", "resnet50")
    checkpoint_path = os.getenv(
        "MODEL_PATH",
        os.path.join(PROJECT_ROOT, "models", "baseline_resnet50_best.pth"),
    )
    device = resolve_device()

    if model_name != "resnet50":
        raise ValueError("The Week 3 API currently serves the ResNet50 checkpoint.")

    model = build_resnet50(num_outputs=len(TARGET_COLS), pretrained=False, dropout=0.0)
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
    return ModelBundle(
        model=model,
        device=device,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        checkpoint_loaded=checkpoint_loaded,
        target_cols=TARGET_COLS,
    )
