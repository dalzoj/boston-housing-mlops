from abc import ABC, abstractmethod

import pandas as pd

from src.core.models import DriftReport


class DriftDetector(ABC):
    @abstractmethod
    def detect(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        pass
