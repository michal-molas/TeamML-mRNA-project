from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm


class Scorer(ABC):
    name: str
    score_names: tuple[str, ...]

    @abstractmethod
    def score(
        self,
        utr5: str,
        cds: str,
        utr3: str,
    ) -> dict[str, float]:
        """Score a single sample based on its sequence components."""
        pass

    def score_df(
        self,
        df: pd.DataFrame,
        index_cols: list[str] = ["id"],
        progress_bar: bool = False,
    ) -> pd.DataFrame:
        """Score a DataFrame of samples, returning a new DataFrame with scores."""
        scores = []

        iterator = df.iterrows()
        if progress_bar:
            iterator = tqdm(iterator, total=len(df), desc="Scoring samples")

        for _, row in iterator:
            score = self.score(
                utr5=row.get("utr5", ""),
                cds=row.get("cds", ""),
                utr3=row.get("utr3", ""),
            )
            scores.append(score)

        scores_df = pd.DataFrame(scores)
        return pd.concat([df[index_cols].reset_index(drop=True), scores_df], axis=1)