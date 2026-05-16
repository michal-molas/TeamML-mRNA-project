from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

try:
    from .schemas import SAMPLE_INDEX_COLS
except ImportError:
    from schemas import SAMPLE_INDEX_COLS


@dataclass
class EvalConfig:
    feature_extractors: list[dict]
    scoring_models: list[dict]
    aggregators: list[dict]
    plots: list[dict]


class FeatureExtractor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def extract_row_features(self, row) -> dict[str, float]:
        raise NotImplementedError()

    def extract_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        index_cols = samples[SAMPLE_INDEX_COLS]
        features = samples.apply(self.extract_row_features, axis=1, result_type="expand")
        return pd.concat([index_cols, features], axis=1)


class ScoringModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score_row(self, row) -> dict[str, float]:
        raise NotImplementedError()

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        index_cols = features[SAMPLE_INDEX_COLS]
        scores = features.apply(self.score_row, axis=1, result_type="expand")
        return pd.concat([index_cols, scores], axis=1)


class Aggregator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, scores: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
