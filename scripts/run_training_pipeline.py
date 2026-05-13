import logging

from src.core.config import get_config
from src.core.logging import setup_logging
from src.data.repository import load_data
from src.pipeline.trainer import train_and_log
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "boston-housing"
MODEL_REGISTRY_NAME = "boston-housing-regressor"


MODEL_CONFIGS = {
    "ridge": {"alpha": 1.0, "random_state": 42},
    "gradient_boosting": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": 42,
    },
}


def main() -> None:
    setup_logging()
    config = get_config()
    logger.info("Iniciando pipeline de entrenamiento con %s", config.mlflow_tracking_uri)

    # Cargar datos
    df = load_data(config.sqlite_path, config.sqlite_data_table_name)
    logger.info("Se han cargado %d elementos desde SQLite", len(df))

    # Asignar MLFlow
    mlflow_client = MLflowClientWrapper()
    experiment_id = mlflow_client.get_or_create_experiment(EXPERIMENT_NAME)

    # Entrenar modelos y recolectar resultados.
    results = []
    for model_name, params in MODEL_CONFIGS.items():
        result = train_and_log(model_name, params, df, experiment_id)
        results.append(result)

    # elegir el ganador respecto RMSE
    winner = min(results, key=lambda r: r["metrics"]["rmse"])
    logger.info(
        "Ganador: %s con RMSE=%.4f (run_id=%s)",
        winner["model_name"], winner["metrics"]["rmse"], winner["run_id"],
    )

    # Registrar modelo
    version = mlflow_client.register_model(
        run_id=winner["run_id"],
        artifact_path="model",
        model_name=MODEL_REGISTRY_NAME,
    )
    mlflow_client.transition_to_staging(MODEL_REGISTRY_NAME, version.version)

    logger.info(
        "Pipeline de entrenamiento completo. Modelo '%s' v%s se encuentra en Stagging",
        MODEL_REGISTRY_NAME, version.version,
    )


if __name__ == "__main__":
    main()
