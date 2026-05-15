# Boston Housing — Pipeline MLOps

Pipeline MLOps end-to-end para predecir el valor medio de viviendas (dataset Boston Housing) servido como **API REST**. La solución es **agnóstica a la nube**: todo el stack es open-source y self-hosted, orquestado con `docker-compose`.

---

## Stack

| Componente | Tecnología |
|---|---|
| API REST | FastAPI + Uvicorn |
| Pipeline ML | scikit-learn (Ridge, GradientBoosting) |
| Tuning | Optuna (TPE) |
| Registry & Tracking | MLflow (self-hosted) |
| Almacenamiento de artefactos / logs | MinIO (S3-compatible) |
| Datos | SQLite |
| Monitoreo | Prometheus + Grafana |
| Drift | Kolmogorov-Smirnov (`scipy.stats.ks_2samp`) |
| Data augmentation | Gaussian Copulas (`copulas`) |
| Config | Pydantic Settings + YAML |
| Contenedores | Docker + docker-compose |
| Dependencias | Poetry |
| Calidad | Ruff, Black, Pytest |
| CI | GitHub Actions |

---

## Estructura del proyecto

```
.
├── config/                    # Configuración YAML (pipeline, training, tuning)
├── data/raw/HousingData.csv   # Dataset original
├── docker/                    # Dockerfiles + init scripts
├── monitoring/                # Configuración Prometheus + Grafana
├── notebooks/                 # EDA + comparativa de modelos
├── scripts/                   # Entry points (entrenamiento, validación, drift, tráfico)
├── src/
│   ├── api/                   # FastAPI: routes, model loader, métricas
│   ├── core/                  # Config, logging, modelos Pydantic
│   ├── data/                  # Repositorio SQLite + schema
│   ├── drift/                 # Detector KS (abstracto + concreto)
│   ├── mlops/                 # Estrategias de training, promoción, retrain, tests de integración
│   ├── pipeline/              # Pipeline sklearn (preprocessor + estimator)
│   ├── storage/               # Clientes MLflow y MinIO
│   └── utils/                 # Logging de predicciones
├── tests/                     # Unit + integration tests (pytest)
├── docker-compose.yml
└── pyproject.toml
```

---

## Requisitos previos

- Docker + Docker Compose
- Python 3.12 + Poetry (solo si se ejecuta fuera de contenedores)

---

## Configuración

Crear un archivo `.env` en la raíz a partir del archivo `.env.example`. Como el siguiente:

```dotenv
# MinIO
MINIO_ROOT_USER=
MINIO_ROOT_PASSWORD=
MINIO_ENDPOINT=http://localhost:9000

# MLFlow
MLFLOW_TRACKING_URI=http://localhost:5001

# Grafana
GF_SECURITY_ADMIN_USER=
GF_SECURITY_ADMIN_PASSWORD=
```

El archivo `config/config.yml` centraliza los parámetros del pipeline (paths, nombres de experimento/registro, umbrales de drift, mejora mínima de RMSE para promoción).

---

## Desarrollo

### 1. Levantar infraestructura

```bash
docker-compose up -d --build
```

Servicios expuestos:

| Servicio | URL |
|---|---|
| API | http://localhost:8000 (docs: `/docs`) |
| MLflow UI | http://localhost:5001 |
| MinIO Console | http://localhost:9001 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

### 2. Instalación de Poetry y dependencias

Instalar Poetry y verificar la instalación:

```bash
pip install poetry
poetry --version
```

Instalar las dependencias del proyecto

```bash
poetry install
```

### 2. Preparar los datosdatos

Migrar el CSV a SQLite. Opcionalmente generar registros sintéticos vía Gaussian Copulas:

```bash
poetry run python -m scripts.migrate_csv_to_sqlite                      # solo migración
poetry run python -m scripts.migrate_csv_to_sqlite --new_elements 500   # + augmentation
```

### 3. Entrenar el modelo

Tres estrategias disponibles (ver sección [Pipeline de entrenamiento](#pipeline-de-entrenamiento)):

```bash
poetry run python -m scripts.run_training_pipeline --type fixed     # hiperparámetros fijos
poetry run python -m scripts.run_training_pipeline --type search    # tuning con Optuna
poetry run python -m scripts.run_training_pipeline --type inherit   # hereda del modelo en producción
```

El ganador (menor RMSE como métrica del proyecto) se registra en MLflow y queda con alias `staging`.

### 4. Validar y promover a producción

```bash
poetry run python -m scripts.validate_and_promote
```

Ejecuta tests de integración sobre el candidato en `staging`. Si pasa, lo promueve a `production`.

### 5. Recargar el modelo en la API

```bash
curl -X POST http://localhost:8000/admin/reload
```

---

## API REST

### Endpoints

| Método | Path | Descripción |
|---|---|---|
| GET | `/health` | Liveness + metadatos del modelo cargado |
| POST | `/predict` | Inferencia |
| POST | `/admin/reload` | Recarga el modelo con alias `production` |
| GET | `/metrics` | Métricas Prometheus |
| GET | `/docs` | Swagger UI |

### Ejemplo

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crim": 0.00632, "zn": 18.0, "indus": 2.31, "chas": 0,
    "nox": 0.538, "rm": 6.575, "age": 65.2, "dis": 4.09,
    "rad": 1, "tax": 296.0, "ptratio": 15.3, "b": 396.9, "lstat": 4.98
  }'
```

Respuesta:

```json
{
  "prediction": 24.18,
  "model_name": "boston-housing-regressor",
  "model_version": "3"
}
```

Cada request válido se loguea en MinIO (`prediction-logs/YYYY/MM/DD/<uuid>.json`) para alimentar el detector de drift.

---

## Pipeline de entrenamiento

El pipeline (`src/pipeline`) está construido sobre la abstracción `BaseRegressionPipeline`: cada modelo concreto (`Ridge`, `GradientBoosting`) solo implementa su estimator. El `Pipeline` de sklearn encadena `StandardScaler` + estimator, garantizando que el preprocesamiento viaje con el modelo serializado.

### Estrategias

| Estrategia | Cuándo usarla | Comportamiento |
|---|---|---|
| `fixed` | Baseline reproducible | Lee hiperparámetros de `config/training_fixed.yml` |
| `search` | Búsqueda inicial / mejora dirigida | Optimización bayesiana con Optuna (espacio en `config/training_search.yml`) |
| `inherit` | Reentrenamiento sobre datos nuevos | Hereda el modelo + hiperparámetros del que esté en `production` |

Todos los runs quedan registrados en MLflow con métricas (`RMSE`, `MAE`, `R²`), tags (`model_name`, `training_strategy`, `dataset_rows`), parámetros y los artefactos YAML de configuración.

---

## Promoción y validación

`scripts.validate_and_promote` ejecuta tres checks sobre el candidato en `staging`:

1. **Deserialización** — el modelo carga correctamente desde MLflow.
2. **Compatibilidad de schema** — predice sobre una muestra real del dataset.
3. **Champion/Challenger** — el challenger debe mejorar el RMSE del modelo en `production` en al menos `min_rmse_improvement` (configurable, default `0.1` miles). Si no hay champion, el challenger se promueve directamente.

Si algún check falla se persiste un `rejection_report.json` como artefacto del run en MLflow.

---

## Monitoreo

La API expone métricas Prometheus en `/metrics`:

| Métrica | Tipo | Descripción |
|---|---|---|
| `predictions_total` | Counter | Predicciones servidas (labels: `model_name`, `model_version`) |
| `prediction_errors_total` | Counter | Errores por tipo (validación, modelo no cargado, fallo de inferencia) |
| `prediction_latency_seconds` | Histogram | Latencia de `/predict` |
| `loaded_model` | Info | Metadatos del modelo activo |

Grafana viene pre-provisionado con el dashboard `api_overview.json` (datasource Prometheus configurado automáticamente).

Para generar tráfico de prueba (con fases de carga normal, burst, quiet y errores):

```bash
poetry run python -m scripts.simulate_traffic           # tráfico normal
poetry run python -m scripts.simulate_traffic --drift   # inyecta drift en ~20% de requests
```

---

## Detección de drift y reentrenamiento

`scripts.lambda_drift` implementa el ciclo completo de drift, pensado para ejecutarse periódicamente (cron, scheduler, lambda equivalente):

1. Carga el dataset de referencia desde SQLite.
2. Descarga las predicciones del día desde MinIO.
3. Aplica **Kolmogorov-Smirnov** por feature (umbral `drift_ks_threshold`).
4. Si detecta drift → llama a `retrain()` (estrategia `inherit`) → valida → promueve si aplica.

```bash
poetry run python -m scripts.lambda_drift
```

---

## Tests

```bash
poetry run pytest tests/ -v
```

- **Unit**: pipeline base, pipelines concretos, detector KS.
- **Integration**: endpoints `/health` y `/predict` con un modelo dummy inyectado.

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`) corre en cada push a `main` y cada PR:

1. Setup Python 3.12 + Poetry.
2. Instalación de dependencias.
3. `ruff format --check` (formato).
4. `ruff check` (lint).
5. `pytest` sobre `tests/`.

---

## Posibles mejoras

- **Job CD**: extender el workflow para construir y publicar imágenes Docker (GHCR) en entornos Cloud
- **Cambiar SQLite como Base de Datos de MLFlow**: Aunque SQLite es suficiente para desarrollo, se ve limitado por concurrencia en producción.
- **Scheduler nativo**: Orquestar `lambda_drift` y `validate_and_promote` con Airflow, Prefect o un cron containerizado en vez de invocación manual.
- **Drift sobre el target**: actualmente solo se detecta covariate drift, es posible agregar concept drift cuando lleguen labels reales. Incluso, complementar el test KS con otras técnicas (PSI, Jensen-Shannon, Wasserstein) para tener una señal más robusta y menos sensible al tamaño de muestra.
- **Alertas en Grafana**: definir reglas sobre p95 de latencia, tasa de errores y drift detectado.
- **Validación de rangos en `PredictionRequest`**: hoy se aceptan floats arbitrarios; añadir `Field(ge=..., le=...)` por feature según percentiles del training set.
- **Modelo más expresivo**: probar XGBoost / LightGBM y un stacking ensemble si la mejora de RMSE lo justifica.

---

## Uso de herramientas AI

Se usó **Claude (Anthropic)** como apoyo durante el desarrollo, principalmente para:

- Agilizar la generación inicial de código boilerplate (Dockerfiles, scaffolding de tests, configuración de Prometheus/Grafana).
- Redacción y revisión de documentación (este README, docstrings, mensajes de log).
- Discusión de decisiones de diseño puntuales (estructura de carpetas, separación de responsabilidades entre `mlops/`, `pipeline/` y `storage/`).

Todas las decisiones de arquitectura, la lógica de negocio (estrategias de entrenamiento, flujo de promoción, detector de drift) y la validación final del código son del autor.
