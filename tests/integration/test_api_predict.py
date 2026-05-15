import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.models import LoadedModel
from src.data.schema import FEATURE_COLUMNS


def _build_dummy_loaded_model() -> LoadedModel:
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(30, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y = rng.rand(30) * 50

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", Ridge()),
        ]
    )
    pipeline.fit(X, y)

    return LoadedModel(pipeline=pipeline, name="test-model", version="1")


import src.api.model_loader as model_loader  # noqa: E402

model_loader._loaded_model = _build_dummy_loaded_model()

from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402

client = TestClient(app)


SAMPLE_PAYLOAD = {
    "crim": 0.00632,
    "zn": 18.0,
    "indus": 2.31,
    "chas": 0,
    "nox": 0.538,
    "rm": 6.575,
    "age": 65.2,
    "dis": 4.09,
    "rad": 1,
    "tax": 296.0,
    "ptratio": 15.3,
    "b": 396.9,
    "lstat": 4.98,
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["code"] == 200
    assert body["content"] == "OK"
    assert body["model_name"] == "test-model"
    assert body["model_version"] == "1"


def test_predict_returns_200_and_expected_shape():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["model_name"] == "test-model"
    assert body["model_version"] == "1"


def test_predict_rejects_missing_field():
    payload = SAMPLE_PAYLOAD.copy()
    payload.pop("crim")

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
