import logging
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.pipeline.base_pipeline import BaseRegressionPipeline

logger = logging.getLogger(__name__)


class RidgePipeline(BaseRegressionPipeline):
    @property
    def model_name(self) -> str:
        return "ridge"

    def _create_estimator(self) -> BaseEstimator:
        return Ridge(**self.model_params)


class GradientBoostingPipeline(BaseRegressionPipeline):
    @property
    def model_name(self) -> str:
        return "gradient_boosting"

    def _create_estimator(self) -> BaseEstimator:
        return GradientBoostingRegressor(**self.model_params)


PIPELINE_REGISTRY: dict[str, type[BaseRegressionPipeline]] = {
    "ridge": RidgePipeline,
    "gradient_boosting": GradientBoostingPipeline,
}


def get_pipeline(name: str, model_params: dict[str, Any] | None = None) -> BaseRegressionPipeline:
    if name not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Pipeline desconocido '{name}'. Disponibles: {list(PIPELINE_REGISTRY.keys())}"
        )
    logger.info("Se ha generado el pipeline de '%s'", name)
    return PIPELINE_REGISTRY[name](model_params=model_params)


def available_pipelines() -> list[str]:
    return list(PIPELINE_REGISTRY.keys())
