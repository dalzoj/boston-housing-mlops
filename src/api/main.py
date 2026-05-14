import logging
from fastapi import FastAPI

from src.api.metrics import set_model_info
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

app.include_router(router)
