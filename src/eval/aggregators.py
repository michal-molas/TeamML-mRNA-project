from eval import Aggregator

import pandas as pd


class SimpleAverageAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def aggregate(self, scores: pd.DataFrame) -> pd.DataFrame:
        # Implement logic to aggregate scores
        # This is a placeholder implementation, replace with actual aggregation logic
        score_cols = [col for col in scores.columns if col not in ["id", "sample"]]
        scores["average_score"] = scores[score_cols].mean(axis=1)
        return scores[["id", "sample", "average_score"]]