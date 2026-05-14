import logging

import pandas as pd
from scipy.stats import ks_2samp

from src.core.config import get_config
from src.core.models import FeatureDriftResult
from src.data.schema import FEATURE_COLUMNS
from src.drift.base import DriftDetector, DriftReport

logger = logging.getLogger(__name__)
config = get_config()


class KSDriftDetector(DriftDetector):

    def detect(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        threshold = config.drift_ks_threshold

        logger.info(
            "Ejecutando detector de KS Drift con Referencia=%d filas, Actual=%d filas, KS Threshold=%.3f",
            len(reference), len(current), threshold,
        )

        per_feature: list[FeatureDriftResult] = []

        for feature in FEATURE_COLUMNS:
            if feature not in reference.columns or feature not in current.columns:
                logger.warning("Feature '%s' ausente en algún dataset", feature)
                continue

            ref_values = reference[feature].dropna()
            cur_values = current[feature].dropna()

            if len(ref_values) < 2 or len(cur_values) < 2:
                logger.warning(
                    "Feature '%s' con muy pocos datos no-NaN (Referente=%d, Actual=%d).",
                    feature, len(ref_values), len(cur_values),
                )
                continue

            statistic, p_value = ks_2samp(ref_values, cur_values)

            per_feature.append(
                FeatureDriftResult(
                    feature=feature,
                    statistic=float(statistic),
                    p_value=float(p_value),
                    drifted=p_value < threshold,
                )
            )

        any_drifted = any(r.drifted for r in per_feature)

        report = DriftReport(
            drifted=any_drifted,
            threshold=threshold,
            per_feature=per_feature,
            n_reference=len(reference),
            n_current=len(current),
        )

        if any_drifted:
            logger.warning(
                "Drift detectado en %d features: %s",
                len(report.drifted_features()), report.drifted_features(),
            )
        else:
            logger.info("No se ha detectado KS Drift")

        return report
