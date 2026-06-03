import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api.main import app


def make_upload():
    image = Image.new("RGB", (32, 32), color=(40, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return {"file": ("sample.jpg", buffer, "image/jpeg")}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "resnet50"
    assert body["num_outputs"] == 37


def test_predict_endpoint(client):
    response = client.post("/predict", files=make_upload())

    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 37
    assert "inference_time_ms" in body


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict",
        files={"file": ("sample.txt", io.BytesIO(b"hello"), "text/plain")},
    )

    assert response.status_code == 400
