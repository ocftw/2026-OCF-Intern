from datasets import load_dataset

try:
    print("Loading test_random split using data_files...")
    dataset_random = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic", data_files="data/test_random-00000-of-00001.parquet")
    print(f"Loaded test_random split: {len(dataset_random['train'])} samples.")
    print("Sample 0 keys:", dataset_random['train'][0].keys())
    print("Sample 0 text:", dataset_random['train'][0]["text"])

    print("\nLoading test_semantic split using data_files...")
    dataset_semantic = load_dataset("ZihCiLin/traditional-chinese-ocr-synthetic", data_files="data/test_semantic-00000-of-00001.parquet")
    print(f"Loaded test_semantic split: {len(dataset_semantic['train'])} samples.")
    print("Sample 0 keys:", dataset_semantic['train'][0].keys())
    print("Sample 0 text:", dataset_semantic['train'][0]["text"])
except Exception as e:
    print("Error:", e)
