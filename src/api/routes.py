import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.metrics import prediction_counter, prediction_errors, prediction_latency
from src.api.model_loader import get_loaded_model, reload_production_model
from src.core.models import HealthResponse, PredictionRequest, PredictionResponse
from src.data.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        return HealthResponse(
            code=200,
            content="OK",
        )
    except Exception as e:
        logger.error("API no funcionando: %s", e)
        return HealthResponse(status=500, content="ERROR")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        loaded = get_loaded_model()
    except RuntimeError as e:
        prediction_errors.labels(error="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail=str(e))

    row = {col: getattr(request, col) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])

    start = time.perf_counter()

    try:
        prediction = float(loaded.pipeline.predict(df)[0])

    except Exception as e:
        prediction_errors.labels(error=type(e).__name__).inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        elapsed = time.perf_counter() - start
        prediction_latency.labels(model_version=loaded.version).observe(elapsed)

    prediction_counter.labels(
        model_name=loaded.name,
        model_version=loaded.version,
    ).inc()

    return PredictionResponse(
        prediction=prediction,
        model_name=loaded.name,
        model_version=loaded.version,
    )


@router.post("/admin/reload")
async def reload():
    loaded = reload_production_model()
    return {"reloaded": True, "version": loaded.version}


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
