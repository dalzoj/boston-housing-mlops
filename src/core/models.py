from pydantic import BaseModel
from typing import Any, List, Optional


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""

# Entrenamiento

class IntegrationReport(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    all_passed: bool
    checks: list[CheckResult] = []


# API

class LoadedModel(BaseModel):
    pipeline: Any
    name: str
    version: str


class HealthResponse(BaseModel):
    code: int
    content: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None


# Predicciones

class PredictionRequest(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: float
    nox: float
    rm: float
    age: float
    dis: float
    rad: float
    tax: float
    ptratio: float
    b: float
    lstat: float

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    model_version: str


# Drift

class FeatureDriftResult(BaseModel):
    feature: str
    statistic: float
    p_value: float
    drifted: bool


class DriftReport(BaseModel):
    drifted: bool
    threshold: float
    per_feature: List[FeatureDriftResult]
    n_reference: int
    n_current: int

    def drifted_features(self) -> list[str]:
        return [
            r.feature
            for r in self.per_feature
            if r.drifted
        ]
