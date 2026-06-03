import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from app.api.inference import predict_image, read_image


class FixedOutputModel(torch.nn.Module):
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        values = torch.linspace(0.0, 1.0, 37)
        return values.repeat(batch_size, 1)


def make_image_bytes():
    image = Image.new("RGB", (32, 32), color=(120, 80, 40))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_read_image_accepts_valid_bytes():
    image = read_image(make_image_bytes())

    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_read_image_rejects_invalid_bytes():
    with pytest.raises(ValueError):
        read_image(b"not an image")


def test_predict_image_returns_probabilities():
    bundle = SimpleNamespace(
        model=FixedOutputModel(),
        device=torch.device("cpu"),
        model_name="fixed",
        target_cols=[f"Class{i}" for i in range(37)],
    )

    result = predict_image(read_image(make_image_bytes()), bundle)

    assert result["model_name"] == "fixed"
    assert len(result["predictions"]) == 37
    assert result["top_prediction"] == "Class36"
    assert result["inference_time_ms"] >= 0
