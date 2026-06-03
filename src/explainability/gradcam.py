import argparse
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.models.model_factory import build_model_from_config  # noqa: E402


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_target_layers(model, model_name: str):
    if model_name in {"resnet18", "resnet50"}:
        return [model.layer4[-1]]
    if model_name == "efficientnet_b0":
        return [model.features[-1]]
    raise ValueError(f"Grad-CAM target layer is not defined for {model_name}")


def build_image_transform(config: dict):
    preproc = config["preprocessing"]
    return transforms.Compose([
        transforms.Resize((preproc["image_size"], preproc["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=preproc["normalize_mean"],
            std=preproc["normalize_std"],
        ),
    ])


def generate_gradcam(
    image_path: str,
    checkpoint_path: str,
    config_path: str,
    output_path: str,
    target_category: int | None = None,
):
    config = load_config(config_path)
    model_name = config["model"].get("name", config["model"].get("architecture", "resnet18"))
    num_outputs = config["model"].get("num_outputs", 37)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = build_model_from_config(config, num_outputs=num_outputs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = build_image_transform(config)
    input_tensor = transform(image).unsqueeze(0).to(device)

    targets = None
    if target_category is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        targets = [ClassifierOutputTarget(target_category)]

    cam = GradCAM(model=model, target_layers=get_target_layers(model, model_name))
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    image_size = config["preprocessing"]["image_size"]
    image_array = np.float32(image.resize((image_size, image_size))) / 255.0
    visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(visualization).save(output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a trained Galaxy Zoo model.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--output_path", default="reports/figures/gradcam.png")
    parser.add_argument("--target_category", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = generate_gradcam(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_path=args.output_path,
        target_category=args.target_category,
    )
    print(f"Saved Grad-CAM to {path}")
