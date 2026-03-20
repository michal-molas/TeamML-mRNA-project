from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import os
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd


DATASETS_DIR = "data/refseq"
DATASETS = [
    "vertebrate_mammalian",
    "vertebrate_other",
]
# fna or gbff
DATA_FORMAT = "fna"

MIN_UTR5_LEN = 10
MIN_CDS_LEN = 50
MIN_UTR3_LEN = 0

MAX_UTR5_LEN = 200
MAX_CDS_LEN = 500
MAX_UTR3_LEN = 0

TRUNCATE_UTR5 = False
TRUNCATE_CDS = False
TRUNCATE_UTR3 = True

CSV_OUTPUT_DIR = "data/pretraining"
CSV_COLUMNS = ["id", "utr5", "cds", "utr3"]
CSV_CHUNK_SIZE = int(1e8) # save CSV chunk after this many records

CHECK_METADATA_DUPLICATES = False


def get_file_paths(dataset_dir: str) -> list[str]:
    """Get all file paths in the dataset directory."""
    file_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(f".{DATA_FORMAT}") or file.endswith(f".{DATA_FORMAT}.gz"):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths)


def _get_seqio_format() -> str:
    if DATA_FORMAT == "gbff":
        return "genbank"
    elif DATA_FORMAT == "fna":
        return "fasta"
    else:
        raise ValueError(f"Unsupported data format: {DATA_FORMAT}")


def _process_file(file_path: str, metadata_df: pd.DataFrame) -> list[dict]:
    """Parse a fasta file and extract transcript sequences and metadata."""
    records = []
    opener = gzip.open if file_path.endswith(".gz") else open
    seqio_format = _get_seqio_format()
    with opener(file_path, "rt") as handle:
        for rec in tqdm(SeqIO.parse(handle, seqio_format), desc=f"Processing {os.path.basename(file_path)}"):
            try:
                metadata = metadata_df.loc[rec.id]
            except KeyError:
                continue

            # Biopython: 0-based start, end-exclusive
            cds_start = int(metadata["utr5_len"])
            cds_end = int(metadata["utr5_len"] + metadata["cds_len"])

            utr5 = rec.seq[:cds_start]
            cds_seq = rec.seq[cds_start:cds_end]
            utr3 = rec.seq[cds_end:]

            if TRUNCATE_UTR5:
                utr5 = utr5[-MAX_UTR5_LEN:]
            if TRUNCATE_CDS:
                cds_seq = cds_seq[:MAX_CDS_LEN]
            if TRUNCATE_UTR3:
                utr3 = utr3[:MAX_UTR3_LEN]

            # Filtering by length
            if not (MIN_UTR5_LEN <= len(utr5) <= MAX_UTR5_LEN):
                continue
            if not (MIN_CDS_LEN <= len(cds_seq) <= MAX_CDS_LEN):
                continue
            if not (MIN_UTR3_LEN <= len(utr3) <= MAX_UTR3_LEN):
                continue

            records.append({
                "id": rec.id,
                "utr5": str(utr5),
                "cds": str(cds_seq),
                "utr3": str(utr3),
            })

    return records


def process_file(file_path: str, metadata_df: pd.DataFrame) -> list[dict]:
    """Wrapper to process a GBFF file with error handling."""
    try:
        return _process_file(file_path, metadata_df)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []  # Return empty list for corrupt files


def save_records(records: list[dict], output_path: str) -> None:
    """Save extracted records to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    part_id = 0
    all_records = []
    corrupt_files = []

    for i, dataset in enumerate(DATASETS):
        print(f"Processing dataset: {dataset} ({i+1}/{len(DATASETS)})")
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        dataset_file_paths = get_file_paths(dataset_dir)
        print(f"Found {len(dataset_file_paths)} files for dataset '{dataset}'")

        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        print(f"Loading metadata from {metadata_path}...")
        metadata_df = pd.read_csv(metadata_path)
        metadata_df.set_index("id", inplace=True)

        for file_path in tqdm(dataset_file_paths, desc=f"Processing files in {dataset}"):
            records = process_file(file_path, metadata_df)
            print(f"  Extracted {len(records)} valid transcripts from {file_path}")

            all_records.extend(records)
            print(f"  {len(all_records)} records extracted so far in total.")

            if len(records) == 0:
                corrupt_files.append(file_path)
                print(f"  No valid records extracted from {file_path}. Marking as potentially corrupt.")

            if len(all_records) >= CSV_CHUNK_SIZE:
                output_path = os.path.join(CSV_OUTPUT_DIR, f"pretraining_data_part{part_id}.csv")
                save_records(all_records, output_path)

                all_records = []
                part_id += 1

    if all_records:
        output_path = os.path.join(CSV_OUTPUT_DIR, f"pretraining_data_part{part_id}.csv")
        save_records(all_records, output_path)

    if corrupt_files:
        print("\nThe following files were potentially corrupt (no valid records extracted):")
        for f in corrupt_files:
            print(f"  {f}")