from app.api.schemas import HealthResponse


def get_health() -> HealthResponse:
    return HealthResponse()
