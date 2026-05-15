import numpy as np
import pandas as pd

from src.data.schema import FEATURE_COLUMNS
from src.drift.ks_detector import KSDriftDetector


def _make_df(seed: int, shift: float = 0.0, n: int = 300) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({col: rng.normal(loc=shift, scale=1.0, size=n) for col in FEATURE_COLUMNS})


def test_no_drift_when_data_is_identical():
    reference = _make_df(seed=42)
    current = reference.copy()

    report = KSDriftDetector().detect(reference, current)

    assert report.drifted is False
    assert report.drifted_features() == []


def test_drift_when_distributions_differ_strongly():
    reference = _make_df(seed=42, shift=0.0)
    current = _make_df(seed=42, shift=5.0)

    report = KSDriftDetector().detect(reference, current)

    assert report.drifted is True
    assert len(report.drifted_features()) == len(FEATURE_COLUMNS)


def test_report_metadata_is_populated():
    reference = _make_df(seed=42, n=100)
    current = _make_df(seed=43, n=80)

    report = KSDriftDetector().detect(reference, current)

    assert report.n_reference == 100
    assert report.n_current == 80
    assert report.threshold == 0.05  # valor que viene del config.yml
    assert len(report.per_feature) == len(FEATURE_COLUMNS)
