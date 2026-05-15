import logging
from typing import Any

from src.core.config import get_config
from src.data.repository import load_data
from src.mlops.training_strategies import inherit_strategy
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)
config = get_config()

MODEL_REGISTRY_NAME = config.mlflow_model_registry_name
EXPERIMENT_NAME = config.mlflow_experiment_name


def retrain() -> dict[str, Any]:

    client = MLflowClientWrapper()

    # Cargar datos actuales desde SQL
    df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)
    logger.info("Reentrenamiento con %d registros (Estrategia: inherit)", len(df))

    # Entrenar y registrar métricas
    experiment_id = client.get_or_create_experiment(EXPERIMENT_NAME)
    results = inherit_strategy(df, experiment_id)
    result = results[0]

    # Registrar nuevo modelo y promoverlo a STAGING
    version = client.register_model(
        run_id=result["run_id"],
        name="model",
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
