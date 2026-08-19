import urllib.request
import os
import sys
import argparse
from pathlib import Path

def download_file(url, filename):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "models"))
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    url_model = "https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/GLM-OCR-Q8_0.gguf"
    file_model = os.path.join(output_dir, "GLM-OCR-Q8_0.gguf")
    if not os.path.exists(file_model):
        download_file(url_model, file_model)
    else:
        print(f"{file_model} already exists.")

    url_proj = "https://huggingface.co/ggml-org/GLM-OCR-GGUF/resolve/main/mmproj-GLM-OCR-Q8_0.gguf"
    file_proj = os.path.join(output_dir, "mmproj-GLM-OCR-Q8_0.gguf")
    if not os.path.exists(file_proj):
        download_file(url_proj, file_proj)
    else:
        print(f"{file_proj} already exists.")
