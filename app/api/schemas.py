from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "galaxy-classification-api"


class ModelInfoResponse(BaseModel):
    model_name: str
    checkpoint_path: str
    checkpoint_loaded: bool
    device: str
    num_outputs: int
    available_models: list[dict]


class PredictionResponse(BaseModel):
    predictions: dict[str, float] = Field(..., description="37 Galaxy Zoo probabilities")
    top_prediction: str
    inference_time_ms: float
    model_name: str
    checkpoint_loaded: bool


class ExplainResponse(PredictionResponse):
    gradcam_png_base64: str
