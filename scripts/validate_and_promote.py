import logging

from src.core.logging import setup_logging
from src.core.config import get_config
from src.mlops.promotion import promote_if_valid


logger = logging.getLogger(__name__)
config = get_config()

MODEL_REGISTRY_NAME = config.mlflow_model_registry_name


def main() -> None:
    setup_logging()
    logger.info("Iniciando proceso de validación y promocion de '%s'", MODEL_REGISTRY_NAME)

    promoted, report = promote_if_valid(MODEL_REGISTRY_NAME)

    if report is None:
        logger.error("Reporte se encuentra vacío.")
        return

    logger.info("Resumen:")
    logger.info(" Modelo:          %s v%s", report.model_name, report.model_version)
    logger.info(" Run ID:          %s", report.run_id)
    logger.info(" Resultado:       %s", "PROMOTED" if promoted else "REJECTED")
    logger.info(" Verificaciones:  %d/%d", sum(1 for c in report.checks if c.passed), len(report.checks))

if __name__ == "__main__":
    main()
