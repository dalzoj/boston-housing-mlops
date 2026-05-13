import logging
from typing import Any
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.pipeline.preprocessor import build_preprocessor

logger = logging.getLogger(__name__)


class BaseRegressionPipeline(ABC):
    def __init__(self, model_params: dict[str, Any] | None = None) -> None:
        self.model_params: dict[str, Any] = model_params or {}

    def build(self) -> Pipeline:
        logger.debug("Construyendo %s pipeline con parámetros=%s", self.model_name, self.model_params)
        return Pipeline(
            steps=[
                ("preprocessor", self._create_preprocessor()),
                ("estimator", self._create_estimator()),
            ]
        )

    def _create_preprocessor(self) -> ColumnTransformer:
        return build_preprocessor()

    @abstractmethod
    def _create_estimator(self) -> BaseEstimator:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
