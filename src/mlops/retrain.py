import logging
from typing import Any

from src.core.config import get_config
from src.data.schema import NON_HYPERPARAMS
from src.data.repository import load_data
from src.pipeline.trainer import train_and_log
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)
config = get_config()

MODEL_REGISTRY_NAME = config.mlflow_model_registry_name
EXPERIMENT_NAME = config.mlflow_experiment_name

def _read_production_config(client: MLflowClientWrapper) -> tuple[str, dict[str, Any]]:

    prod = client.get_version_by_alias(MODEL_REGISTRY_NAME, "production")

    if prod is None:
        raise RuntimeError(
            f"No existe modelo con 'production' para '{MODEL_REGISTRY_NAME}'"
        )

    run = client.get_run(prod.run_id)

    model_name = run.data.tags.get("model_name")

    hyperparams = {}
    for key, raw_value in run.data.params.items():
        if key in NON_HYPERPARAMS:
            continue
        hyperparams[key] = _parse_param_value(raw_value)

    logger.info(
        "Se han heredado los hiperpárametros de %s v%s: model_name=%s, params=%s",
        MODEL_REGISTRY_NAME, prod.version, model_name, hyperparams,
    )

    return model_name, hyperparams


def _parse_param_value(raw: str) -> Any:
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except (ValueError, TypeError):
        return raw


def retrain() -> dict[str, Any]:
    client = MLflowClientWrapper()

    # Leer los hiperparámetros del modelo en prodicción
    model_name, hyperparams = _read_production_config(client)

    # Cargar datos actuales desde SQL
    df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)
    logger.info("Realizando reentrenamiento con %d registros", len(df))

    # Entrenar y registrar métricas
    experiment_id = client.get_or_create_experiment(EXPERIMENT_NAME)
    result = train_and_log(model_name, hyperparams, df, experiment_id)

    # Registrar nuevo modelo y promoverlo a STAGING
    version = client.register_model(
        run_id=result["run_id"],
        artifact_path="model",
        model_name=MODEL_REGISTRY_NAME,
    )
    client.promote_to_staging(MODEL_REGISTRY_NAME, version.version)

    logger.info(
        "Modelo reentrenado registrado como v%s y promovido a Staging",
        version.version,
    )

    return {
        **result,
        "new_version": version.version,
    }
