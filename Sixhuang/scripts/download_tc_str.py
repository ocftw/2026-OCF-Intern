import urllib.request
import tarfile
import os
import sys
import argparse
from pathlib import Path

def download_and_extract(url, filename, extract_path):
    if not os.path.exists(filename):
        print(f"Downloading {url} to {filename}...")
        def report(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = min(100, read_so_far * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent:.2f}% ({read_so_far}/{total_size} bytes)")
            else:
                sys.stdout.write(f"\rRead {read_so_far} bytes")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, reporthook=report)
        print("\nDownload finished.")
    else:
        print(f"{filename} already exists, skipping download.")

    print(f"Extracting {filename} to {extract_path}...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(path=extract_path)
    print("Extraction finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "data"))
    args = parser.parse_args()
    url = "https://rd-tcsynth.swlab.cloud/download?filename=TC-STR.tar.gz"
    dest_dir = os.path.abspath(args.output_dir)
    os.makedirs(dest_dir, exist_ok=True)
    archive = os.path.join(dest_dir, "TC-STR.tar.gz")
    download_and_extract(url, archive, dest_dir)
