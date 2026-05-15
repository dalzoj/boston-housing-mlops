import logging
import time

import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.metrics import prediction_counter, prediction_errors, prediction_latency
from src.api.model_loader import get_loaded_model, reload_production_model
from src.core.models import HealthResponse, PredictionRequest, PredictionResponse
from src.data.schema import FEATURE_COLUMNS
from src.utils.utils import log_prediction

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    try:
        loaded = get_loaded_model()
        return HealthResponse(
            code=200,
            content="OK",
            model_name=loaded.name,
            model_version=loaded.version,
        )

    except Exception as e:
        logger.error("API no funciona: %s", e)
        return HealthResponse(
            code=500,
            content="ERROR",
            model_name=None,
            model_version=None,
        )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> PredictionResponse:
    try:
        loaded = get_loaded_model()
    except Exception as e:
        prediction_errors.labels(error="model_not_loaded").inc()
        logger.error("No se pudo cargar el modelo: %s", e)
        raise HTTPException(status_code=503, detail=str(e))

    row = {col: getattr(request, col) for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row])

    start = time.perf_counter()

    try:
        prediction = float(loaded.pipeline.predict(df)[0])

    except Exception as e:
        prediction_errors.labels(error="prediction_failed").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        elapsed = time.perf_counter() - start
        prediction_latency.labels(model_version=loaded.version).observe(elapsed)

    background_tasks.add_task(log_prediction, row, prediction, loaded.version)


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
