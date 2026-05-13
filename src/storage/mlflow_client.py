import logging
import os
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from src.core.config import get_config

logger = logging.getLogger(__name__)


class MLflowClientWrapper:
    def __init__(self) -> None:
        config = get_config()

        os.environ["AWS_ACCESS_KEY_ID"] = config.minio_root_user
        os.environ["AWS_SECRET_ACCESS_KEY"] = config.minio_root_password
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = config.minio_endpoint

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=config.mlflow_tracking_uri)
        logger.debug("MLFlow inicializado en %s", config.mlflow_tracking_uri)

    def health_check(self) -> bool:
        try:
            self._client.search_experiments(max_results=1)
            return True
        except Exception as e:
            logger.warning("MLFlow no se encuentra en línea: %s", e)
            return False

    def get_or_create_experiment(self, name: str) -> str:
        experiment = self._client.get_experiment_by_name(name)
        if experiment is None:
            exp_id = self._client.create_experiment(name)
            logger.info("Creando experimento MLFlow en '%s' (id=%s)", name, exp_id)
            return exp_id
        return experiment.experiment_id

    def list_experiment_names(self) -> list[str]:
        return [e.name for e in self._client.search_experiments()]

    def register_model(
        self,
        run_id: str,
        artifact_path: str,
        model_name: str,
    ) -> ModelVersion:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        version = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(
            "Registro de modelo '%s' version %s del run %s",
            model_name, version.version, run_id,
        )
        return version

    def transition_to_staging(self, model_name: str, version: str) -> None:
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        logger.info("Modelo '%s' v%s promovido a STAGING", model_name, version)

    def transition_to_production(self, model_name: str, version: str) -> None:
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Modelo '%s' v%s promovido a PRODUCTION", model_name, version)

    def get_latest_version(self, model_name: str, stage: str) -> ModelVersion | None:
        versions = self._client.get_latest_versions(model_name, stages=[stage])
        return versions[0] if versions else None

    def best_run_in_experiment(
        self,
        experiment_id: str,
        metric: str = "rmse",
        ascending: bool = True,
    ) -> dict[str, Any] | None:
        order = "ASC" if ascending else "DESC"
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        if not runs:
            return None
        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "model_name": run.data.tags.get("model_name", "unknown"),
            metric: run.data.metrics.get(metric),
        }
