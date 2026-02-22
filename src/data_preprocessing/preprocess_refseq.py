from Bio import SeqIO
import gzip
from pathlib import Path
import pandas as pd
from tqdm import main, tqdm

GBFF_DIR = Path("data/refseq")


def _get_gbff_paths() -> list[Path]:
    gbff_paths = list(GBFF_DIR.glob("*.gbff"))
    gbff_gz_paths = list(GBFF_DIR.glob("*.gbff.gz"))
    return gbff_paths + gbff_gz_paths


if __name__ == "__main__":
    # 1. Find all transcripts and their associated metadata files
    print("Loading reference sequences...")
    all_paths = _get_gbff_paths()

    print(f"Found {len(all_paths)} GBFF files:")
    for path in all_paths:
        print(f"  {path}")

    # 2. Parse each GBFF file and extract transcript sequences and metadata
    # Split into 5', 3' UTRs and CDS
    print("Parsing GBFF files and extracting sequences...")

    records = []

    for path in all_paths:
        print(f"Processing {path}...")

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for rec in tqdm(SeqIO.parse(handle, "genbank"), desc=f"Parsing {path.name}"):

                cds_feats = [ft for ft in rec.features if ft.type == "CDS"]
                if not cds_feats:
                    continue  # noncoding (NR_) / pseudogene / etc.

                # usually exactly one CDS per transcript record
                cds = cds_feats[0]

                # Biopython: 0-based start, end-exclusive
                cds_start0 = int(cds.location.start)
                cds_end0   = int(cds.location.end)

                utr5  = rec.seq[:cds_start0]
                cds_seq   = cds.extract(rec.seq)   # robust to join/complement
                utr3 = rec.seq[cds_end0:]

                # Convert to GenBank-style (1-based inclusive) coords on transcript:
                cds_begin_1 = cds_start0 + 1
                cds_end_1   = cds_end0

                acc = rec.id  # e.g., NM_000123.4

                records.append({
                    "utr5": str(utr5),
                    "cds": str(cds_seq),
                    "utr3": str(utr3),
                    "accession": acc,
                    "source_file": path.name,
                })

    records_df = pd.DataFrame(records)

    # TODO: Define filtering to exclude too long, duplicate etc seqs

    print(f"Extracted {len(records_df)} transcript records.")
    print(records_df.head())

    # 3. Save to CSV
    output_path = Path("data/refseq/refseq_transcripts.csv")
    records_df.to_csv(output_path, index=False)
    print(f"Saved transcript records to {output_path}")
