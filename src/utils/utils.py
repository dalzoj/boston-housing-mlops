import uuid
import logging
from datetime import datetime, timezone

from src.storage.minio_client import MinioClient
from src.core.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

def log_prediction(features: dict, prediction: float, model_version: str) -> None:
    try:

        minio = MinioClient()

        now = datetime.now(timezone.utc)
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
