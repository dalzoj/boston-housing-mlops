import json
import logging
from pathlib import Path
from dataclasses import asdict

import mlflow
import mlflow.sklearn
import pandas as pd

from src.core.config import get_config
from src.data.repository import load_data
from src.data.schema import FEATURE_COLUMNS
from src.core.models import IntegrationReport, CheckResult

logger = logging.getLogger(__name__)


SAMPLE_SIZE = 50


def _check_deserialization(model_uri: str) -> tuple[dict, object | None]:
    try:
        model = mlflow.sklearn.load_model(model_uri)

        return (
            CheckResult(name="deserialization", passed=True, detail="Modelo cargado correctamente"),
            model,
        )

    except Exception as e:
        return (
            CheckResult(name="deserialization", passed=False, detail=str(e)),
            None,
        )


def _check_schema(model, sample_df: pd.DataFrame) -> dict:
    try:
        X = sample_df[FEATURE_COLUMNS].head(SAMPLE_SIZE)
        _ = model.predict(X.head(3))
        return CheckResult(
            name="schema_compatibility",
            passed=True,
            detail="Esquema compatible",
        )

    except Exception as e:
        return CheckResult(
            name="schema_compatibility",
            passed=False,
            detail=str(e),
        )

def run_integration_tests(model_name: str, version: str, run_id: str) -> IntegrationReport:
    config = get_config()
    model_uri = f"models:/{model_name}/{version}"

    logger.info("Corriendo test de integración de %s v%s (run_id=%s)", model_name, version, run_id)

    report = IntegrationReport(
        model_name=model_name,
        model_version=version,
        run_id=run_id,
        all_passed=False,
    )

    # Validar carga del modelo
    deser_result, model = _check_deserialization(model_uri)
    report.checks.append(deser_result)
    if not deser_result.passed:
        logger.error("Carga fallida. Finalizando proceso de test de integración.")
        return report

    # Validar esquema de entrada
    df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)
    schema_result = _check_schema(model, df)
    report.checks.append(schema_result)

    report.all_passed = all(c.passed for c in report.checks)

    for check in report.checks:
        status = "YES" if check.passed else "NO"
        logger.info( "  [%s] %s: %s", status, check.name, check.detail)

    return report


def save_rejection_report(report: IntegrationReport, run_id: str) -> Path:
    rejection_path = Path(f"/tmp/rejection_report_{run_id}.json")
    rejection_path.write_text(json.dumps(asdict(report), indent=2))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(rejection_path), artifact_path="validation")

    logger.warning("Reporte de expulsión MLFLow. (run_id: %s)", run_id)
    return rejection_path
