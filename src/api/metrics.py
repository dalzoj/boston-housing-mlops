from prometheus_client import Counter, Histogram, Info

prediction_counter = Counter(
    "predictions_check_total",
    "Número total de predicciones atendidas",
    labelnames=["model_name", "model_version"],
)

prediction_errors = Counter(
    "prediction_errors_total",
    "Número total de errores de predicción",
    labelnames=["error"],
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Latencia de /predict en segundos",
    labelnames=["model_version"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

model_info = Info(
    "loaded_model",
    "Metadatos del modelo cargado actualmente",
)


def set_model_info(name: str, version: str) -> None:
    model_info.info({"name": name, "version": version})
