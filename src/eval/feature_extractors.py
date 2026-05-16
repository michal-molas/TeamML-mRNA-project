from eval import FeatureExtractor

import pandas as pd


class StringStatisticsExtractor(FeatureExtractor):
    def __init__(self):
        super().__init__()

    def extract_row_features(self, row) -> dict[str, float]:
        return {
            "cds_length": len(row["cds"]),
            "utr5_length": len(row["utr5"]),
            "utr3_length": len(row["utr3"]),
        }

    def extract_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        # Implement logic to extract features from the samples
        # This is a placeholder implementation, replace with actual feature extraction logic

        # Sanitize input
        samples = samples.fillna("")

        return pd.DataFrame({
            "id": samples["id"],
            "sample": samples["sample"],
            "cds_length": samples["cds"].apply(len),
            "utr5_length": samples["utr5"].apply(len),
            "utr3_length": samples["utr3"].apply(len),
        })