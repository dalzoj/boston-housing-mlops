import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error

from src.core.config import get_config
from src.data.schema import NON_HYPERPARAMS
from src.pipeline.concrete_pipeline import get_pipeline
from src.pipeline.trainer import split_data, train_and_log
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)
config = get_config()

FIXED_CONFIG_PATH = Path("config/training_fixed.yml")
SEARCH_CONFIG_PATH = Path("config/training_search.yml")


def inherit_strategy(df: pd.DataFrame, experiment_id: str) -> list[dict[str, Any]]:
    client = MLflowClientWrapper()
    prod = client.get_version_by_alias(config.mlflow_model_registry_name, "production")

    if prod is None:
        raise RuntimeError("No hay modelo en 'production' para heredar")

    run = client.get_run(prod.run_id)
    model_name = run.data.tags.get("model_name")
    hyperparams = _coerce_mlflow_params(run.data.params)

    logger.info("Heredando de v%s: model=%s, params=%s", prod.version, model_name, hyperparams)

    # Guardar hiperparámetros en YAML temporalmente
    inherited_yaml = _dump_inherited_yaml(model_name, hyperparams, prod.version)

    result = train_and_log(
        model_name=model_name,
        model_params=hyperparams,
        df=df,
        experiment_id=experiment_id,
        extra_params={
            "training_strategy": "inherit",
            "inherited_from_version": prod.version,
        },
        extra_artifacts=[inherited_yaml],
    )
    return [result]


def _dump_inherited_yaml(model_name: str, hyperparams: dict, parent_version: str) -> Path:

    path = Path(tempfile.gettempdir()) / "inherited_params.yml"
    content = {"models": {model_name: hyperparams}}
    path.write_text(yaml.safe_dump(content, sort_keys=False))

    return path


def fixed_strategy(df: pd.DataFrame, experiment_id: str) -> list[dict[str, Any]]:

    hiperparams_config = yaml.safe_load(FIXED_CONFIG_PATH.read_text())

    logger.info("Estrategia fixed: %d modelos", len(hiperparams_config["models"]))

    return [
        train_and_log(
            model_name=name,
            model_params=params,
            df=df,
            experiment_id=experiment_id,
            extra_params={"training_strategy": "fixed"},
            extra_artifacts=[FIXED_CONFIG_PATH],
        )
        for name, params in hiperparams_config["models"].items()
    ]


def search_strategy(df: pd.DataFrame, experiment_id: str) -> list[dict[str, Any]]:

    hiperparams_config = yaml.safe_load(SEARCH_CONFIG_PATH.read_text())

    logger.info(
        "Estrategia search: %d modelos, %d trials, seed=%d",
        len(hiperparams_config["models"]), hiperparams_config["n_trials"], hiperparams_config["seed"],
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results = []
    for model_name, space in hiperparams_config["models"].items():
        best_params = _run_optuna(model_name, space, df, hiperparams_config["n_trials"], hiperparams_config["seed"])
        result = train_and_log(
            model_name=model_name,
            model_params=best_params,
            df=df,
            experiment_id=experiment_id,
            extra_params={
                "training_strategy": "search",
                "optuna_n_trials": hiperparams_config["n_trials"],
                "optuna_seed": hiperparams_config["seed"],
            },
            extra_artifacts=[SEARCH_CONFIG_PATH],
        )
        results.append(result)
    return results


def _run_optuna(model_name, space, df, n_trials, seed):

    X_train, X_test, y_train, y_test = split_data(df)

    def objective(trial):
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        pipeline = get_pipeline(model_name, params).build()
        pipeline.fit(X_train, y_train)
        return float(np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test))))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**study.best_params, **{k: v["value"] for k, v in space.items() if v["type"] == "fixed"}}
    logger.info("Mejor %s: RMSE=%.4f, params=%s", model_name, study.best_value, best)
    return best


def _suggest(trial, name, spec):

    if spec["type"] == "fixed":
        return spec["value"]

    if spec["type"] == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))

    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def _coerce_mlflow_params(params: dict[str, str]) -> dict[str, Any]:
    out = {}

    # Convertir al tipo real del hiperparámetro
    for key, raw in params.items():
        if key in NON_HYPERPARAMS:
            continue
        try:
            out[key] = float(raw) if "." in raw else int(raw)
        except (ValueError, TypeError):
            out[key] = raw

    return out


STRATEGY_REGISTRY = {
    "inherit": inherit_strategy,
    "fixed": fixed_strategy,
    "search": search_strategy,
}
