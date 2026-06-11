#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import tqdm
import RNA

def clean_seq(x: object) -> str:
    """Return uppercase RNA sequence. Empty/NaN becomes ''."""
    if pd.isna(x):
        return ""
    return str(x).strip().upper().replace("T", "U")


def fold(seq: str):
    """ViennaRNA fold wrapper. Returns (structure, mfe)."""
    if not seq:
        return "", float("nan")
    structure, mfe = RNA.fold(seq)
    return structure, float(mfe)


def aug_window(utr5: str, cds: str, upstream: int, downstream: int) -> str:
    """
    Window around start codon:
    last `upstream` nt of UTR5 + first `downstream` nt of CDS.

    If CDS starts with AUG, the AUG is naturally included in cds[:downstream].
    """
    return utr5[-upstream:] + cds[:downstream]


def process_row(row_dict, upstream: int, downstream: int):
    row_id = row_dict.get("id", "")
    utr5 = clean_seq(row_dict.get("utr5", ""))
    cds = clean_seq(row_dict.get("cds", ""))
    utr3 = clean_seq(row_dict.get("utr3", ""))
    split = row_dict.get("split", "")

    sequence = utr5 + cds + utr3
    aug_seq = aug_window(utr5, cds, upstream, downstream)

    structure, mfe = fold(sequence)
    # utr5_structure, utr5_mfe = fold(utr5)
    # cds_structure, cds_mfe = fold(cds)
    # utr3_structure, utr3_mfe = fold(utr3)
    # aug_structure, aug_mfe = fold(aug_seq)

    return {
        "id": row_id,
        "utr5": utr5,
        "cds": cds,
        "utr3": utr3,
        "split": split,
        "sequence": sequence,
        "structure": structure,
        "mfe": mfe,
        # "utr5_structure": utr5_structure,
        # "utr5_mfe": utr5_mfe,
        # "cds_structure": cds_structure,
        # "cds_mfe": cds_mfe,
        # "utr3_structure": utr3_structure,
        # "utr3_mfe": utr3_mfe,
        # "aug_window_sequence": aug_seq,
        # "aug_window_structure": aug_structure,
        # "aug_window_mfe": aug_mfe,
        "seq_len": len(sequence),
        "utr5_len": len(utr5),
        "cds_len": len(cds),
        "utr3_len": len(utr3),
        "aug_window_len": len(aug_seq),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Add ViennaRNA MFE and dot-bracket structure columns to mRNA CSV."
    )
    parser.add_argument("--input", required=True, help="Input CSV with columns: id, utr5, cds, utr3")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--workers", type=int, default=1, help="CPU worker processes")
    parser.add_argument("--aug_upstream", type=int, default=50, help="nt taken from end of UTR5")
    parser.add_argument("--aug_downstream", type=int, default=100, help="nt taken from beginning of CDS")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"id", "utr5", "cds", "utr3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = df.to_dict(orient="records")

    if args.workers <= 1:
        out_rows = [process_row(r, args.aug_upstream, args.aug_downstream) for r in rows]
    else:
        out_rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_row, r, args.aug_upstream, args.aug_downstream) for r in rows]

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Folding RNA"):
                out_rows.append(future.result())

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output, index=False)
    print(f"Saved {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
