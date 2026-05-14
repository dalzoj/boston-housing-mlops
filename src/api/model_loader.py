import logging
import threading

import mlflow.sklearn

from src.core.config import get_config
from src.core.models import LoadedModel
from src.storage.mlflow_client import MLflowClientWrapper


logger = logging.getLogger(__name__)
config = get_config()

MODEL_REGISTRY_NAME = config.mlflow_model_registry_name
PRODUCTION_ALIAS = "production"


_lock = threading.Lock()
_loaded_model: LoadedModel | None = None


def load_production_model() -> LoadedModel:
    client = MLflowClientWrapper()
    version_info = client.get_version_by_alias(MODEL_REGISTRY_NAME, PRODUCTION_ALIAS)

    if version_info is None:
        raise RuntimeError(
            f"No se encontró ningún modelo con el alias '{PRODUCTION_ALIAS}' para '{MODEL_REGISTRY_NAME}'"
        )

    model_uri = f"models:/{MODEL_REGISTRY_NAME}@{PRODUCTION_ALIAS}"
    logger.info("Cargando modelo de %s", model_uri)
    pipeline = mlflow.sklearn.load_model(model_uri)

    loaded_model = LoadedModel(
        pipeline=pipeline,
        name=MODEL_REGISTRY_NAME,
        version=version_info.version,
    )
    logger.info("Cargado modelo %s v%s", loaded_model.name, loaded_model.version)
    return loaded_model


def reload_production_model() -> LoadedModel:
    global _loaded_model
    with _lock:
        _loaded_model = load_production_model()
    return _loaded_model


def get_loaded_model() -> LoadedModel:
    global _loaded_model
    if _loaded_model is None:
        with _lock:
            if _loaded_model is None:
                _loaded_model = load_production_model()
    return _loaded_model
