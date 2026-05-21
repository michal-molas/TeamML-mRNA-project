try:
    from .schemas import SAMPLE_INDEX_COLS
except ImportError:
    from schemas import SAMPLE_INDEX_COLS

import pandas as pd


def string_statistics_extractor(samples: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id": samples["id"],
        "sample": samples["sample"],
        "cds_length": samples["cds"].apply(len),
        "utr5_length": samples["utr5"].apply(len),
        "utr3_length": samples["utr3"].apply(len),
    })


def gc_content(seq: str) -> float:
    gc_count = seq.count("G") + seq.count("C")
    total_count = len(seq)
    return gc_count / total_count if total_count > 0 else 0.0


def gc_content_extractor(samples: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "id": samples["id"],
        "sample": samples["sample"],
        "cds_gc_content": samples["cds"].apply(gc_content),
        "utr5_gc_content": samples["utr5"].apply(gc_content),
        "utr3_gc_content": samples["utr3"].apply(gc_content),
    })