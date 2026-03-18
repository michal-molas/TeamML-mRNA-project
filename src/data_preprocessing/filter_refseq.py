import csv
import gzip
import os
from Bio import SeqIO
from tqdm import tqdm


DATASETS_DIR = "data/refseq"
DATASETS = [
    "vertebrate_mammalian",
    "vertebrate_other",
]

MIN_UTR5_LEN = 10
MIN_CDS_LEN = 50
MIN_UTR3_LEN = 10

MAX_UTR5_LEN = 200
MAX_CDS_LEN = 3000
MAX_UTR3_LEN = 3000

TRUNCATE_UTR5 = False
TRUNCATE_CDS = False
TRUNCATE_UTR3 = False

CSV_OUTPUT_DIR = "data/pretraining"
CSV_COLUMNS = ["id", "utr5", "cds", "utr3"]
CSV_CHUNK_SIZE = 100000


def get_gbff_paths(dataset_dir: str) -> list[str]:
    """Get all GBFF file paths in the dataset directory."""
    gbff_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".gbff") or file.endswith(".gbff.gz"):
                gbff_paths.append(os.path.join(root, file))
    return sorted(gbff_paths)


def _process_file(file_path: str) -> list[dict]:
    """Parse a GBFF file and extract transcript sequences and metadata."""
    records = []
    opener = gzip.open if file_path.endswith(".gz") else open
    with opener(file_path, "rt") as handle:
        for rec in tqdm(SeqIO.parse(handle, "genbank")):
            cds_feats = [ft for ft in rec.features if ft.type == "CDS"]
            if not cds_feats:
                continue  # noncoding (NR_) / pseudogene / etc.

            # usually exactly one CDS per transcript record
            cds = cds_feats[0]

            # Biopython: 0-based start, end-exclusive
            cds_start0 = int(cds.location.start)
            cds_end0   = int(cds.location.end)

            utr5 = rec.seq[:cds_start0]
            cds_seq = cds.extract(rec.seq)
            utr3 = rec.seq[cds_end0:]

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


def process_file(file_path: str) -> list[dict]:
    """Wrapper to process a GBFF file with error handling."""
    try:
        return _process_file(file_path)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gbff_paths = []
    for dataset in DATASETS:
        dataset_dir = os.path.join(DATASETS_DIR, dataset)
        dataset_gbff_paths = get_gbff_paths(dataset_dir)
        print(f"Found {len(dataset_gbff_paths)} GBFF files for dataset '{dataset}'")
        gbff_paths.extend(dataset_gbff_paths)

    print(f"Total GBFF files found: {len(gbff_paths)}")

    part_id = 0
    all_records = []

    for gbff_path in gbff_paths:
        print(f"Processing {gbff_path}...")
        records = process_file(gbff_path)
        print(f"  Extracted {len(records)} valid transcripts from {gbff_path}")

        all_records.extend(records)

        # Save chunked output to avoid memory issues
        if len(all_records) >= 100000:
            output_path = os.path.join(OUTPUT_DIR, f"pretraining_data_part{part_id}.csv")
            save_records(all_records, output_path)

            all_records = []
            part_id += 1

    # Save any remaining records
    if all_records:
        output_path = os.path.join(OUTPUT_DIR, f"pretraining_data_part{part_id}.csv")
        save_records(all_records, output_path)
