import logging
import uuid
from datetime import UTC, datetime

from src.core.config import get_config
from src.storage.minio_client import MinioClient

logger = logging.getLogger(__name__)
config = get_config()


def log_prediction(features: dict, prediction: float, model_version: str) -> None:
    try:
        minio = MinioClient()

        now = datetime.now(UTC)
        payload = {
            "timestamp": now.isoformat(),
            "model_version": model_version,
            "prediction": prediction,
            **features,
        }

        key = f"{now.strftime('%Y/%m/%d')}/{uuid.uuid4()}.json"
        minio.upload_json(
            bucket=config.minio_bucket_prediction_logs,
            key=key,
            payload=payload,
        )
    except Exception as e:
        logger.warning("Failed to log prediction to MinIO: %s", e)
