import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.metrics import prediction_errors, set_model_info
from src.api.model_loader import get_loaded_model
from src.api.routes import router
from src.core.logging import setup_logging

logger = logging.getLogger(__name__)

setup_logging()
logger.info("Iniciando API de Boston Housing")


try:
    loaded = get_loaded_model()
    set_model_info(loaded.name, loaded.version)

    logger.info(
        "Modelo cargado: %s v%s",
        loaded.name,
        loaded.version,
    )


except Exception as e:
    logger.error("Error cargando el modelo al iniciar: %s", e)


app = FastAPI(
    title="Boston Housing Prediction API",
    description="REST API para predecir el valor medio de viviendas en Boston.",
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    prediction_errors.labels(error="validation_failed").inc()
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    prediction_errors.labels(error="internal_error").inc()
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(router)
