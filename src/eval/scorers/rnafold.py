import RNA
import pandas as pd

from .base import Scorer


class RNAfoldScorer(Scorer):
    name = "RNAfold"
    score_names = (
        "rnafold_mfe",
        "rnafold_mfe_per_nt",
    )

    def score(
        self,
        utr5: str,
        cds: str,
        utr3: str,
    ) -> dict[str, float]:
        seq = utr5 + cds + utr3
        _, mfe = RNA.fold(seq)
        return {
            "rnafold_mfe": float(mfe),
            "rnafold_mfe_per_nt": float(mfe) / len(seq) if len(seq) > 0 else 0.0,
        }
