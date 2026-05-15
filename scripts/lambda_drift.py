import logging
from datetime import datetime, timezone

from src.core.config import get_config
from src.core.logging import setup_logging
from src.data.repository import load_data
from src.drift.ks_detector import KSDriftDetector
from src.mlops.promotion import promote_if_valid
from src.mlops.retrain import retrain
from src.storage.minio_client import MinioClient

logger = logging.getLogger(__name__)
config = get_config()
setup_logging()


def main():

    logger.info("Iniciando ciclo de detección de drift...")

    # Cargar el conjunto de referencia
    reference_df = load_data(str(config.sqlite_path), config.sqlite_data_table_name)
    logger.info("Dataset con %d filas", len(reference_df))

    # Descargar las predicciones diarias
    minio = MinioClient()
    today = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    current_df = minio.download_jsons_as_dataframe(
        bucket=config.minio_bucket_prediction_logs,
        prefix=today,
    )

    logger.info("Se han recolectado %d predicciones. Fecha %s", len(current_df), today)

    # Detectar drift
    detector = KSDriftDetector()
    report = detector.detect(reference_df, current_df)

    if not report.drifted:
        logger.info("No se ha detectado drift")
        return

    logger.warning("Se ha detectado drift en: %s", report.drifted_features())

    # Reentrenamiento
    logger.info("Iniciando reentrenamiento")
    retrain_result = retrain()
    logger.info(
        "Modelo reentrenado: v%s con RMSE=%.4f",
        retrain_result["new_version"], retrain_result["metrics"]["rmse"],
    )

    # Validar y promover
    logger.info("Iniciando pruebas de integración y promoción")
    promoted, _ = promote_if_valid(config.mlflow_model_registry_name)

    if promoted:
        logger.info("Se ha promovido un nuevo modelo a producción.")
    else:
        logger.error("No se ha promovido el nuevo modelo a producción.")


if __name__ == "__main__":
    main()
