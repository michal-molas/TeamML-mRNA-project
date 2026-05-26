import pandas as pd

from scorers.base import Scorer


class StringStatisticsScorer(Scorer):
    name = "string_statistics"
    score_names = (
        "utr5_length",
        "cds_length",
        "utr3_length",
        "total_length",
        "utr5_gc_content",
        "cds_gc_content",
        "utr3_gc_content",
    )

    def _gc_content(self, seq: str) -> float:
        if len(seq) == 0:
            return 0.0
        gc_count = seq.count("G") + seq.count("C")
        return gc_count / len(seq)

    def score(
        self,
        utr5: str,
        cds: str,
        utr3: str,
    ) -> dict[str, float]:
        """Score a single sample based on its sequence components."""
        return {
            "utr5_length": len(utr5),
            "cds_length": len(cds),
            "utr3_length": len(utr3),
            "total_length": len(utr5) + len(cds) + len(utr3),
            "utr5_gc_content": self._gc_content(utr5),
            "cds_gc_content": self._gc_content(cds),
            "utr3_gc_content": self._gc_content(utr3),
        }

    def score_df(
        self,
        df: pd.DataFrame,
        index_cols: list[str] = ["id"],
        progress_bar: bool = False,
    ) -> pd.DataFrame:
        return pd.DataFrame({
            "id": df["id"],
            "utr5_length": df["utr5"].str.len(),
            "cds_length": df["cds"].str.len(),
            "utr3_length": df["utr3"].str.len(),
            "total_length": df["utr5"].str.len() + df["cds"].str.len() + df["utr3"].str.len(),
            "utr5_gc_content": df["utr5"].apply(self._gc_content),
            "cds_gc_content": df["cds"].apply(self._gc_content),
            "utr3_gc_content": df["utr3"].apply(self._gc_content),
        })