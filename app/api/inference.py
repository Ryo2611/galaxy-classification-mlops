import base64
import io
import time

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

from app.api.model_loader import ModelBundle


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def read_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc


def predict_image(image: Image.Image, bundle: ModelBundle) -> dict:
    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(bundle.device)

    start = time.perf_counter()
    with torch.no_grad():
        output = bundle.model(input_tensor)[0].detach().cpu().numpy()
    elapsed_ms = (time.perf_counter() - start) * 1000

    predictions = {
        name: round(float(prob), 4)
        for name, prob in zip(bundle.target_cols, output)
    }
    top_prediction = max(predictions, key=predictions.get)
    return {
        "predictions": predictions,
        "top_prediction": top_prediction,
        "inference_time_ms": round(elapsed_ms, 3),
        "model_name": bundle.model_name,
        "checkpoint_loaded": getattr(bundle, "checkpoint_loaded", False),
    }


def generate_gradcam_base64(image: Image.Image, bundle: ModelBundle) -> str:
    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(bundle.device)
    if bundle.model_name in {"resnet18", "resnet50"}:
        target_layers = [bundle.model.layer4[-1]]
    elif bundle.model_name == "efficientnet_b0":
        target_layers = [bundle.model.features[-1]]
    elif bundle.model_name == "convnext_tiny":
        target_layers = [bundle.model.features[-1]]
    else:
        raise ValueError(f"Grad-CAM is not supported for {bundle.model_name}.")

    cam = GradCAM(model=bundle.model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    image_array = np.float32(image.resize((224, 224))) / 255.0
    visualization = show_cam_on_image(image_array, grayscale_cam, use_rgb=True)

    buffer = io.BytesIO()
    Image.fromarray(visualization).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
