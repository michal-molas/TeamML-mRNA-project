from pathlib import Path

import pandas as pd


RIBONN_DATA_DIR = "data/finetuning/ribonn"
RIBONN_HUMAN_DATA_PATH = Path(RIBONN_DATA_DIR) / "ribonn_human.xlsx"
RIBONN_MOUSE_DATA_PATH = Path(RIBONN_DATA_DIR) / "ribonn_mouse.xlsx"

RIBONN_TE_COL = "mean_te" # discuss: which TE to use?
CSV_OUTPUT_DIR = "data/finetuning/ribonn/dataset.csv"


def split_seq(
    seq: str, utr5_size: int, cds_size: int, utr3_size: int
) -> tuple[str, str, str]:
    assert len(seq) == utr5_size + cds_size + utr3_size, "Sequence length does not match sum of sizes"

    utr5 = seq[:utr5_size]
    cds = seq[utr5_size : utr5_size + cds_size]
    utr3 = seq[utr5_size + cds_size : utr5_size + cds_size + utr3_size]
    return utr5, cds, utr3


def preprocess_data(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataframe with (id, utr5, cds, utr3) columns from original dataframe
    with (id, tx_sequence, utr5_size, cds_size, utr3_size).
    """
    required_cols = ["tx_sequence", "utr5_size", "cds_size", "utr3_size"]

    missing = [col for col in required_cols if col not in data_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = []
    for row in data_df.itertuples(index=False):
        utr5, cds, utr3 = split_seq(
            seq=row.tx_sequence,
            utr5_size=row.utr5_size,
            cds_size=row.cds_size,
            utr3_size=row.utr3_size,
        )
        records.append(
            {
                "utr5": utr5,
                "cds": cds,
                "utr3": utr3,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    human_data_df = pd.read_excel(RIBONN_HUMAN_DATA_PATH)
    mouse_data_df = pd.read_excel(RIBONN_MOUSE_DATA_PATH)
    data_df = pd.concat([human_data_df, mouse_data_df], ignore_index=True)

    # use split_seq to create new columns utr5, cds, utr3
    processed_df = preprocess_data(data_df)

    processed_df["id"] = data_df["tx_id"]
    processed_df["te"] = data_df[RIBONN_TE_COL] # normalize TE values to [0, 1] range?

    print(processed_df.head())

    processed_df.to_csv(CSV_OUTPUT_DIR, index=False)
    print(f"Saved processed dataset to {CSV_OUTPUT_DIR}")