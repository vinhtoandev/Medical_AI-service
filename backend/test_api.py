"""
Script test nhanh API bằng requests.
Cài: pip install requests
Chạy: python test_api.py --image path/to/image.jpg
"""

import requests
import json
import argparse
import base64
import os


API_URL = "http://localhost:5000"


def test_health():
    print("=== Health Check ===")
    r = requests.get(f"{API_URL}/health")
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    print()


def test_predict(image_path: str, save_mask: bool = False):
    print(f"=== Predict: {image_path} ===")

    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
        r = requests.post(f"{API_URL}/predict", files=files)

    if r.status_code != 200:
        print(f"Lỗi {r.status_code}: {r.text}")
        return

    result = r.json()

    # In kết quả (bỏ mask để gọn)
    display = {k: v for k, v in result.items() if k != "mask_base64"}
    print(json.dumps(display, indent=2, ensure_ascii=False))

    # Lưu mask nếu có
    if save_mask and result.get("mask_base64"):
        mask_path = image_path.rsplit(".", 1)[0] + "_mask.png"
        with open(mask_path, "wb") as f:
            f.write(base64.b64decode(result["mask_base64"]))
        print(f"\n✓ Đã lưu mask: {mask_path}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn file ảnh")
    parser.add_argument("--save-mask", action="store_true", help="Lưu ảnh mask ra file")
    args = parser.parse_args()

    test_health()
    test_predict(args.image, save_mask=args.save_mask)
