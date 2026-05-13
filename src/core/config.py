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
        case_sensitive=False,
    )

    print("si")
    # MinIO
    minio_root_user: str
    minio_root_password: str

    # SQLite
    sqlite_path: Path
    sqlite_data_table_name: str

    # CSV
    csv_path: Path


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
