import logging

from src.core.models import IntegrationReport
from src.mlops.integration import run_integration_tests, save_rejection_report
from src.storage.mlflow_client import MLflowClientWrapper

logger = logging.getLogger(__name__)


def promote_if_valid(model_name: str) -> tuple[bool, IntegrationReport | None]:
    client = MLflowClientWrapper()

    # Buscar candidato
    staging_version = client.get_version_by_alias(model_name, "staging")
    if staging_version is None:
        logger.warning("No se encuentra modelo en STAGING en '%s'.", model_name)
        return False, None

    logger.info(
        "Modelo candidato encontrado: %s v%s (run_id=%s)",
        model_name,
        staging_version.version,
        staging_version.run_id,
    )

    # Correr el test de integración
    report = run_integration_tests(
        model_name=model_name,
        version=staging_version.version,
        run_id=staging_version.run_id,
    )

    # Promover o expulsar
    if report.all_passed:
        client.promote_to_production(model_name, staging_version.version)
        return True, report

    save_rejection_report(report, staging_version.run_id)
    logger.error(
        "Modelo %s v%s expulsado. Ver detalles (run_id: %s).",
        model_name,
        staging_version.version,
        staging_version.run_id,
    )
    return False, report
