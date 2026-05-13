#!/bin/sh
set -e

echo "Inicializando MinIO..."

mc alias set local http://minio:9000 "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}"

echo "Creando buckets..."
mc mb --ignore-existing "local/mlflow"
mc mb --ignore-existing "local/prediction-logs"

echo "Buckets creados:"
mc ls local/

echo "Inicialización de MinIO correcta."
