from datasets import load_dataset
import sys

try:
    print("Loading test_semantic split...")
    dataset_semantic = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic", split="test_semantic")
    print(f"Loaded test_semantic split: {len(dataset_semantic)} samples.")
    print("Sample 0 keys:", dataset_semantic[0].keys())
    print("Sample 0 text:", dataset_semantic[0]["text"])

    print("Loading test_random split...")
    dataset_random = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic", split="test_random")
    print(f"Loaded test_random split: {len(dataset_random)} samples.")
    print("Sample 0 text:", dataset_random[0]["text"])

except Exception as e:
    print("Error:", e)
