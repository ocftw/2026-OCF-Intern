import requests
import base64
import json
import os
import sys
import argparse
from pathlib import Path

def test_model(model_name, image_path, prompt):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_image]
            }
        ],
        "options": {
            "temperature": 0.0,
            "top_p": 0.00001,
            "top_k": 1
        },
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=str(Path(__file__).resolve().parents[1] / "data" / "TC-STR" / "images" / "billboard_00000_010_雜貨舖.jpg"))
    args = parser.parse_args()
    img_path = args.image
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        sys.exit(1)

    prompts = [
        "Text Recognition:",
        "Please recognize the text in the image. Return only the recognized text, nothing else.",
        "請辨識圖片中的文字。只輸出辨識出的文字，不要有任何其他贅字、標點、說明或格式。",
        "OCR"
    ]

    models = ["chandra-ocr-2", "ggml-org/GLM-OCR-GGUF/GLM-OCR-Q8_0.gguf"]

    for model in models:
        print(f"\n================ Model: {model} ================")
        for prompt in prompts:
            print(f"Prompt: {repr(prompt)}")
            output = test_model(model, img_path, prompt)
            print(f"Output: {repr(output)}")
