# Skin Analysis Pipeline API

Flask REST API kết hợp 3 model AI phân tích da:
1. **Model 1** – Binary classification: skin / not skin
2. **Model 2** – Segmentation: phát hiện vùng tổn thương
3. **Model 3** – Classification: phân loại loại tổn thương

---

## Cấu trúc project

```
skin_pipeline_api/
├── app.py              # Flask app, route định nghĩa
├── pipeline.py         # Logic 3 bước pipeline
├── requirements.txt
├── test_api.py         # Script test nhanh
└── models/             # Thư mục chứa file .h5
    ├── skin_classifier.h5
    ├── segmentation.h5
    └── lesion_classifier.h5
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cấu hình trước khi chạy

Mở `pipeline.py` và chỉnh các giá trị sau:

```python
# Nhãn của model 3 (thêm/bớt theo số class của bạn)
LESION_CLASSES = ["Melanoma", "Nevus", ...]

# Kích thước input của từng model
self._resize(pil_img, (224, 224))   # Model 1
self._resize(pil_img, (256, 256))   # Model 2
self._resize(pil_img, (224, 224))   # Model 3

# Ngưỡng quyết định
SKIN_THRESHOLD      = 0.5    # Model 1
MASK_AREA_THRESHOLD = 0.01   # Model 2 (1% diện tích)
```

---

## Chạy server

### Development
```bash
python app.py
```

### Production (dùng Gunicorn)
```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
> `-w 2`: 2 worker (nên = số CPU). Không đặt quá cao vì mỗi worker load lại model tốn RAM.

---

## API Endpoints

### GET /health
Kiểm tra server và model đã load chưa.

**Response:**
```json
{ "status": "ok", "models_loaded": true }
```

---

### POST /predict
Phân tích ảnh da qua pipeline 3 model.

**Request:** `multipart/form-data`
- `image` (file): ảnh JPG/PNG/BMP/WEBP, tối đa 10MB

**Response cases:**

**Case 1 – Không phải da:**
```json
{
  "stage": "not_skin",
  "is_skin": false,
  "skin_confidence": 0.12,
  "has_lesion": false,
  "lesion_class": null,
  "confidence": null,
  "top_predictions": null,
  "mask_base64": null,
  "message": "Ảnh không chứa vùng da. Pipeline dừng ở bước 1."
}
```

**Case 2 – Da bình thường:**
```json
{
  "stage": "normal_skin",
  "is_skin": true,
  "skin_confidence": 0.95,
  "has_lesion": false,
  "lesion_ratio": 0.002,
  "lesion_class": null,
  "confidence": null,
  "top_predictions": null,
  "mask_base64": "<base64 PNG string>",
  "message": "Da bình thường. Không phát hiện tổn thương."
}
```

**Case 3 – Có tổn thương:**
```json
{
  "stage": "lesion_found",
  "is_skin": true,
  "skin_confidence": 0.98,
  "has_lesion": true,
  "lesion_ratio": 0.15,
  "lesion_class": "Melanoma",
  "confidence": 0.87,
  "top_predictions": [
    {"class": "Melanoma",  "confidence": 0.87},
    {"class": "Nevus",     "confidence": 0.09},
    {"class": "Basal Cell Carcinoma", "confidence": 0.03}
  ],
  "mask_base64": "<base64 PNG string>",
  "message": "Phát hiện tổn thương: Melanoma (87.0%)"
}
```

---

## Test

```bash
# Test với ảnh bất kỳ
python test_api.py --image path/to/image.jpg

# Test và lưu mask ra file
python test_api.py --image path/to/image.jpg --save-mask
```

---

## Lưu ý

- Server load cả 3 model khi khởi động → lần đầu chạy sẽ chậm (~10-30s tùy model)
- Mỗi Gunicorn worker load model riêng → RAM = số_worker × tổng_RAM_3_model
- Nếu model nặng (>500MB), cân nhắc dùng `preload_app = True` trong Gunicorn config
