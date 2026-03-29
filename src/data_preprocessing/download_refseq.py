from pathlib import Path
import re
import requests
from tqdm import tqdm

SAVE_DIR = Path("data/refseq")
REFSEQ_URLS = {
    "vertebrate_mammalian": "https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_mammalian/",
    "vertebrate_other": "https://ftp.ncbi.nlm.nih.gov/refseq/release/vertebrate_other/",
}
# gbff or fna
DATA_FORMAT = "fna"


def get_files_urls(dataset_url: str) -> list[str]:
    response = requests.get(dataset_url)
    text = response.text

    file_urls = []
    for line in text.splitlines():
        if f"rna.{DATA_FORMAT}" in line:
            match = re.search(r'href="([^"]+)"', line)
            if match:
                filename = match.group(1)
                full_url = dataset_url + filename
                file_urls.append(full_url)

    return file_urls


def download_file(file_url: str, save_path: Path) -> None:
    """Download a large file with a progress bar."""
    response = requests.get(file_url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as f, tqdm(
        desc=save_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            f.write(data)
            bar.update(len(data))


if __name__ == "__main__":
    print("Selected data format:", DATA_FORMAT)

    for dataset, dataset_url in REFSEQ_URLS.items():
        print(f"Processing dataset: {dataset}")
        file_urls = get_files_urls(dataset_url)
        print(f"  Found {len(file_urls)} files for {dataset}")

        for file_url in tqdm(file_urls, desc=f"Downloading files for {dataset}"):
            filename = file_url.split("/")[-1]
            save_path = SAVE_DIR / dataset / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists before downloading
            if save_path.exists():
                print(f"  {save_path} already exists, skipping download.")
                continue

            print(f"Downloading {file_url} to {save_path}...")
            download_file(file_url, save_path)
