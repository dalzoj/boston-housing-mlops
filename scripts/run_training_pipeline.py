import argparse
import logging

from src.core.config import get_config
from src.core.logging import setup_logging
from src.data.repository import load_data
from src.mlops.training_strategies import STRATEGY_REGISTRY
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)
config = get_config()

EXPERIMENT_NAME = config.mlflow_experiment_name
MODEL_REGISTRY_NAME = config.mlflow_model_registry_name


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento. Tres estrategias para elegir hiperparámetros")
    parser.add_argument(
        "--type",
        choices=list(STRATEGY_REGISTRY.keys()),
        default="inherit",
        help=("Entrenamiento de tipos: inherit, fixed o search"),
    )
    args = parser.parse_args()

    logger.info("Iniciando pipeline de entrenamiento (Estrategia=%s) en %s", args.type, config.mlflow_tracking_uri)

    # Cargar datos
    df = load_data(config.sqlite_path, config.sqlite_data_table_name)
    logger.info("Se han cargado %d elementos desde SQLite", len(df))

    # Obtener experimento
    mlflow_client = MLflowClientWrapper()
    experiment_id = mlflow_client.get_or_create_experiment(EXPERIMENT_NAME)

    # Ejecutar la estrategia elegida
    strategy = STRATEGY_REGISTRY[args.type]
    results = strategy(df, experiment_id)

    # Elegir el ganador global por RMSE
    winner = min(results, key=lambda r: r["metrics"]["rmse"])
    logger.info(
        "Ganador: %s con RMSE=%.4f (run_id=%s)",
        winner["model_name"], winner["metrics"]["rmse"], winner["run_id"],
    )

    # Registrar y promover a Staging
    version = mlflow_client.register_model(
        run_id=winner["run_id"],
        artifact_path="model",
        model_name=MODEL_REGISTRY_NAME,
    )
    mlflow_client.promote_to_staging(MODEL_REGISTRY_NAME, version.version)

    logger.info(
        "Pipeline completo. Modelo '%s' v%s en STAGING",
        MODEL_REGISTRY_NAME, version.version,
    )


if __name__ == "__main__":
    main()
