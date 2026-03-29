import gzip
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
from Bio import SeqIO
from tqdm import tqdm
import pandas as pd


DATASET_DIRS = {
    "vertebrate_mammalian": "data/refseq/vertebrate_mammalian",
    "vertebrate_other": "data/refseq/vertebrate_other",
}

METADATA_DIRS = {
    "vertebrate_mammalian": "data/refseq/vertebrate_mammalian/metadata",
    "vertebrate_other": "data/refseq/vertebrate_other/metadata",
}


def get_gbff_files(dataset_dir: str) -> list[str]:
    gbff_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".gbff.gz"):
                gbff_files.append(os.path.join(root, file))
    return sorted(gbff_files)


def extract_utr_cds_info(record):
    """Extract CDS and UTR info from a GenBank record."""
    seq_len = len(record.seq)
    cds_start = cds_end = None

    # get organism name
    organism = record.annotations.get("organism", None)


    for feature in record.features:
        if feature.type == "CDS":
            cds_start = int(feature.location.start)  # 0-based
            cds_end = int(feature.location.end)       # 0-based, exclusive
            gene = feature.qualifiers.get("gene", [None])[0]
            break  # take first CDS

    if cds_start is None or cds_end is None:
        return None  # skip records without CDS

    utr5_len = cds_start
    utr3_len = seq_len - cds_end
    cds_len = cds_end - cds_start

    return {
        "id": record.id,
        "organism": organism,
        "gene": gene,
        "seq_len": seq_len,
        "utr5_len": utr5_len,
        "cds_len": cds_len,
        "utr3_len": utr3_len,
    }


def _process_file(file_path: str) -> tuple[str, pd.DataFrame]:
    # check if target already exists
    target_path = f"data/refseq/vertebrate_mammalian/metadata/{os.path.basename(file_path)}.csv"
    if os.path.exists(target_path):
        print(f"Metadata for {file_path} already exists at {target_path}, skipping.")
        return file_path, pd.read_csv(target_path)

    # check if file is known to be corrupt
    # if file_path in corrupt_files:
    #     raise ValueError(f"File {file_path} is known to be corrupt")

    print(f"Worker {os.getpid()} processing {file_path}...")
    rows = []
    with gzip.open(file_path, "rt") as handle:
        for record in tqdm(SeqIO.parse(handle, "genbank")):
            row = extract_utr_cds_info(record)
            if row is not None:
                rows.append(row)
    return file_path, pd.DataFrame(rows)


def process_file(file_path: str) -> tuple[str, pd.DataFrame | None]:
    try:
        return _process_file(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return file_path, None


if __name__ == "__main__":



    for dataset_name, dataset_dir in DATASET_DIRS.items():
        print(f"Processing dataset: {dataset_name}")
        metadata_dir = METADATA_DIRS[dataset_name]
        gbff_files = get_gbff_files(dataset_dir)
        print(f"Found {len(gbff_files)} GBFF files to process.")
        corrupt_files_path = f"{dataset_dir}/corrupt_files.txt"

        save_paths = [f"{metadata_dir}/{os.path.basename(fp)}.csv" for fp in gbff_files]
        existing_files = [p for p in save_paths if os.path.exists(p)]
        if existing_files:
            print(f"Found {len(existing_files)} existing metadata files, skipping these:")
            for p in existing_files:
                print(f"  {p}")
            gbff_files = [fp for fp in gbff_files if f"{metadata_dir}/{os.path.basename(fp)}.csv" not in existing_files]
            print(f"{len(gbff_files)} files remain to process after skipping existing metadata.")

        with ProcessPoolExecutor() as ex:
            for file_path, df in tqdm(ex.map(process_file, gbff_files[::-1]), total=len(gbff_files), desc="GBFF files"):

                save_path = f"{metadata_dir}/{os.path.basename(file_path)}.csv"

                if os.path.exists(save_path):
                    print(f"Metadata for {file_path} already exists at {save_path}, skipping save.")
                    continue

                if df is not None:
                    df.to_csv(f"{metadata_dir}/{os.path.basename(file_path)}.csv", index=False)
                    print(f"Saved metadata for {file_path} with {len(df)} records.")
                else:
                    # write to corrupt files list
                    with open(corrupt_files_path, "a") as f:
                        f.write(file_path + "\n")
                    print(f"Marked {file_path} as corrupt.")

