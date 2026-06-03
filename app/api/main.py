from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.health import get_health
from app.api.inference import generate_gradcam_base64, predict_image, read_image
from app.api.model_loader import ModelBundle, load_model
from app.api.schemas import ExplainResponse, HealthResponse, ModelInfoResponse, PredictionResponse


model_bundle: ModelBundle | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_bundle
    model_bundle = load_model()
    yield


app = FastAPI(
    title="Galaxy Classification API",
    description="Predict Galaxy Zoo morphology probabilities from uploaded galaxy images.",
    version="2.0.0",
    lifespan=lifespan,
)


def get_model_bundle() -> ModelBundle:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return model_bundle


async def read_upload_image(file: UploadFile):
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    contents = await file.read()
    try:
        return read_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_model=HealthResponse)
def root():
    return get_health()


@app.get("/health", response_model=HealthResponse)
def health():
    return get_health()


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    bundle = get_model_bundle()
    return ModelInfoResponse(
        model_name=bundle.model_name,
        checkpoint_path=bundle.checkpoint_path,
        checkpoint_loaded=bundle.checkpoint_loaded,
        device=str(bundle.device),
        num_outputs=len(bundle.target_cols),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image = await read_upload_image(file)
    return PredictionResponse(**predict_image(image, get_model_bundle()))


@app.post("/predict/", response_model=PredictionResponse)
async def predict_legacy(file: UploadFile = File(...)):
    return await predict(file)


@app.post("/predict/explain", response_model=ExplainResponse)
async def predict_explain(file: UploadFile = File(...)):
    image = await read_upload_image(file)
    bundle = get_model_bundle()
    result = predict_image(image, bundle)
    result["gradcam_png_base64"] = generate_gradcam_base64(image, bundle)
    return ExplainResponse(**result)
