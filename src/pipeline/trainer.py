import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN
from src.pipeline.concrete_pipeline import get_pipeline
from src.core.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

RANDOM_STATE = config.random_state
TEST_SIZE = config.test_size

def split_data(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }
    logger.info("Evaluation: %s", metrics)
    return metrics


def train_and_log(
    model_name: str,
    model_params: dict[str, Any],
    df: pd.DataFrame,
    experiment_id: str,
    extra_params: dict[str, Any] | None = None,
    extra_artifacts: list[Path] | None = None,
) -> dict[str, Any]:
    logger.info("Entrenando %s con parámetros=%s", model_name, model_params)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline_obj = get_pipeline(model_name, model_params)
    pipeline = pipeline_obj.build()

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag("model_name", pipeline_obj.model_name)
        mlflow.set_tag("dataset_rows", len(df))
        mlflow.log_params(model_params)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", TEST_SIZE)

        if extra_params:
            mlflow.log_params(extra_params)

        pipeline.fit(X_train, y_train)

        input_example = X_train.iloc[:3]
        y_pred_example = pd.Series(pipeline.predict(input_example), name=TARGET_COLUMN)
        signature = infer_signature(input_example, y_pred_example)

        metrics = evaluate(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train.iloc[:3],
            signature=signature,
        )

        if extra_artifacts:
            for path in extra_artifacts:
                mlflow.log_artifact(str(path))

        run_id = run.info.run_id

    logger.info("Run %s finalizado. RMSE=%.4f", run_id, metrics["rmse"])
    return {
        "run_id": run_id,
        "model_name": pipeline_obj.model_name,
        "metrics": metrics,
    }
