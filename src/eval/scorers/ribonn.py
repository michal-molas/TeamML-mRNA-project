import pandas as pd

from scorers.base import Scorer


class RiboNNScorer(Scorer):
    name = "RiboNN"
    score_names = ("ribonn_te",)

    def score(
        self,
        utr5: str,
        cds: str,
        utr3: str,
    ) -> dict[str, float]:
        return {
            "ribonn_te": 0.5
        }

    def score_df(
        self,
        df: pd.DataFrame,
        index_cols: list[str] = ["id"],
        progress_bar: bool = False,
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "id": df["id"],
            "ribonn_te": 0.5,
        })