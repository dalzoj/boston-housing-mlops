import yaml
import threading
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_lock = threading.Lock()
_instance: "AppConfig | None" = None


class AppConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MinIO
    minio_root_user: str
    minio_root_password: str
    minio_endpoint: str = "http://localhost:9000"
    minio_bucket_prediction_logs: str

    # SQLite
    sqlite_path: Path
    sqlite_data_table_name: str

    # CSV
    csv_path: Path

    # Pipeline
    random_state: int
    test_size: float

    # MLFlow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str
    mlflow_model_registry_name: str

    # Others
    drift_ks_threshold: float = 0.05
    min_rmse_improvement: float = 0.1


def _load_yaml() -> dict:
    with open("config/config.yml", "r") as f:
        return yaml.safe_load(f)


def get_config() -> AppConfig:
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:
                yaml_data = _load_yaml()
                _instance = AppConfig(**yaml_data)

    return _instance
