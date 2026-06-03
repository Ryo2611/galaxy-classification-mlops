import base64
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL.rstrip('/')}/predict"
EXPLAIN_URL = f"{API_BASE_URL.rstrip('/')}/predict/explain"
SAMPLE_DIR = PROJECT_ROOT / "data/processed/rgb_images"


st.set_page_config(
    page_title="Galaxy Classifier XAI",
    page_icon="🌌",
    layout="wide",
)

st.title("🌌 Galaxy Classification & Explainable AI")
st.caption("Upload a galaxy image to inspect morphology probabilities, inference latency, and Grad-CAM.")


def sample_images(limit: int = 20):
    if not SAMPLE_DIR.exists():
        return []
    return sorted(SAMPLE_DIR.glob("*.jpg"))[:limit]


def post_image(image_name: str, image_bytes: bytes, explain: bool):
    url = EXPLAIN_URL if explain else PREDICT_URL
    files = {"file": (image_name, image_bytes, "image/jpeg")}
    response = requests.post(url, files=files, timeout=120)
    response.raise_for_status()
    return response.json()


def render_predictions(result: dict):
    top_prediction = result["top_prediction"]
    inference_time_ms = result["inference_time_ms"]
    model_name = result["model_name"]
    predictions = result["predictions"]

    metric_cols = st.columns(3)
    metric_cols[0].metric("Top class", top_prediction)
    metric_cols[1].metric("Inference", f"{inference_time_ms:.1f} ms")
    metric_cols[2].metric("Model", model_name)

    pred_df = (
        pd.DataFrame(
            [{"class": key, "probability": value} for key, value in predictions.items()]
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Top 10 probabilities")
    st.bar_chart(pred_df.head(10), x="class", y="probability", height=320)

    with st.expander("All 37 probabilities"):
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

    if "gradcam_png_base64" in result:
        st.subheader("Grad-CAM")
        gradcam_bytes = base64.b64decode(result["gradcam_png_base64"])
        st.image(gradcam_bytes, caption="Model attention heatmap", use_container_width=True)


with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Image source", ["Upload", "Sample"], horizontal=True)
    explain = st.checkbox("Generate Grad-CAM", value=True)

    uploaded_file = None
    sample_path = None
    if input_mode == "Upload":
        uploaded_file = st.file_uploader("Galaxy image", type=["png", "jpg", "jpeg"])
    else:
        samples = sample_images()
        if samples:
            sample_label = st.selectbox("Sample image", [path.name for path in samples])
            sample_path = next(path for path in samples if path.name == sample_label)
        else:
            st.info("No sample images found in data/processed/rgb_images.")


try:
    health = requests.get(f"{API_BASE_URL.rstrip('/')}/health", timeout=5)
    health.raise_for_status()
except requests.RequestException:
    st.warning("Backend API is not reachable. Start it with `uvicorn app.api.main:app --reload --port 8000`.")


image_name = None
image_bytes = None
preview_image = None

if uploaded_file is not None:
    image_name = uploaded_file.name
    image_bytes = uploaded_file.getvalue()
    preview_image = Image.open(uploaded_file).convert("RGB")
elif sample_path is not None:
    image_name = sample_path.name
    image_bytes = sample_path.read_bytes()
    preview_image = Image.open(sample_path).convert("RGB")


if image_bytes is None:
    st.info("Choose an upload or sample image to run inference.")
else:
    col_image, col_result = st.columns([1, 1.35])
    with col_image:
        st.subheader("Input image")
        st.image(preview_image, use_container_width=True)

    with col_result:
        if st.button("Run inference", type="primary"):
            with st.spinner("Running model inference..."):
                try:
                    result = post_image(image_name, image_bytes, explain=explain)
                    render_predictions(result)
                except requests.HTTPError as exc:
                    st.error(f"API error: {exc.response.status_code} - {exc.response.text}")
                except requests.RequestException as exc:
                    st.error(f"Could not connect to API: {exc}")
