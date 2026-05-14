from typing import Any
from pydantic import BaseModel


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class IntegrationReport(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    all_passed: bool
    checks: list[CheckResult] = []


class LoadedModel(BaseModel):
    pipeline: Any
    name: str
    version: str


class HealthResponse(BaseModel):
    code: int
    content: str


class PredictionRequest(BaseModel):
    crim: float | None = None
    zn: float | None = None
    indus: float | None = None
    chas: float | None = None
    nox: float | None = None
    rm: float | None = None
    age: float | None = None
    dis: float | None = None
    rad: float | None = None
    tax: float | None = None
    ptratio: float | None = None
    b: float | None = None
    lstat: float | None = None

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
