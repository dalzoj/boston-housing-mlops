import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.schema import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def build_preprocessor() -> ColumnTransformer:
    logger.debug("Construyendo preprocesador para %d características", len(FEATURE_COLUMNS))

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, FEATURE_COLUMNS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
