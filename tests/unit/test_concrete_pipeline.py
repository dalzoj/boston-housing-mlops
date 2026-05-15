import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.pipeline.concrete_pipeline import (
    GradientBoostingPipeline,
    RidgePipeline,
    available_pipelines,
    get_pipeline,
)


def test_available_pipelines_contains_known_models():
    names = available_pipelines()
    assert "ridge" in names
    assert "gradient_boosting" in names


def test_get_pipeline_ridge_returns_correct_type_and_estimator():
    obj = get_pipeline("ridge", {"alpha": 0.5})
    assert isinstance(obj, RidgePipeline)

    built = obj.build()
    assert isinstance(built, Pipeline)
    assert isinstance(built.named_steps["estimator"], Ridge)
    assert built.named_steps["estimator"].alpha == 0.5


def test_get_pipeline_gradient_boosting_returns_correct_type_and_estimator():
    obj = get_pipeline("gradient_boosting", {"n_estimators": 10})
    assert isinstance(obj, GradientBoostingPipeline)

    built = obj.build()
    assert isinstance(built.named_steps["estimator"], GradientBoostingRegressor)
    assert built.named_steps["estimator"].n_estimators == 10


def test_get_pipeline_unknown_name_raises_value_error():
    with pytest.raises(ValueError):
        get_pipeline("unknown_model")
