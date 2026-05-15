import json
import logging
from typing import Any

import boto3
import pandas as pd
from botocore.client import Config

from src.core.config import get_config

logger = logging.getLogger(__name__)
config = get_config()


class MinioClient:
    def __init__(self) -> None:

        self._client = boto3.client(
            "s3",
            endpoint_url=config.minio_endpoint,
            aws_access_key_id=config.minio_root_user,
            aws_secret_access_key=config.minio_root_password,
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )

        logger.debug("MinioClient inicializado en %s", config.minio_endpoint)

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:

        keys: list[str] = []

        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])

        return keys

    def download_json(self, bucket: str, key: str) -> dict[str, Any]:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())

    def download_jsons_as_dataframe(self, bucket: str, prefix: str) -> pd.DataFrame:

        keys = self.list_objects(bucket, prefix)

        if not keys:
            return pd.DataFrame()

        records = []

        for key in keys:
            try:
                records.append(self.download_json(bucket, key))
            except Exception as e:
                logger.warning("No se pudo leer %s/%s: %s", bucket, key, e)

        logger.info(
            "Descargados %d registros de predicciones desde %s/%s",
            len(records),
            bucket,
            prefix,
        )
        return pd.DataFrame(records)

    def upload_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self._client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        logger.debug("Se ha cargado JSON a %s/%s ", bucket, key)
