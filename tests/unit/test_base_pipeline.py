import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.pipeline.base_pipeline import BaseRegressionPipeline


class DummyPipeline(BaseRegressionPipeline):
    @property
    def model_name(self) -> str:
        return "dummy"

    def _create_estimator(self):
        return Ridge()


def test_base_pipeline_is_abstract():
    with pytest.raises(TypeError):
        BaseRegressionPipeline()


def test_build_returns_sklearn_pipeline_with_expected_steps():
    pipeline = DummyPipeline().build()

    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "estimator" in pipeline.named_steps


def test_default_model_params_is_empty_dict():
    dummy = DummyPipeline()
    assert dummy.model_params == {}


def test_model_params_are_stored():
    params = {"alpha": 2.5}
    dummy = DummyPipeline(model_params=params)
    assert dummy.model_params == params
