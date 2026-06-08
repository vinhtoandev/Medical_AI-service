from itertools import chain
import numpy as np

from flask import Flask, request, jsonify
from pipeline import SkinAnalysisPipeline, META_FEATURES_ORDER
import os
from flask_cors import CORS
from chatbot import answer_question
from huggingface_hub import hf_hub_download


app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:8080"]}},
    supports_credentials=True,
)
# Khởi tạo pipeline (load model một lần khi start server)
classifier_path = hf_hub_download(
    repo_id="toanle1355/skin_binary",
    filename="mobilenetv2_skin_classifier.h5",
    repo_type="model",
)
segmentation_path = hf_hub_download(
    repo_id="toanle1355/skin_binary",
    filename="unet.h5",
    repo_type="model",
) 
lesion_classifier_path = hf_hub_download(
    repo_id="toanle1355/multi-modal_fusion",
    filename="model_resnet50_best (1).keras",
    repo_type="model",
)

# lesion_classifier_path = "model/model_resnet50_best (1).keras"

pipeline = SkinAnalysisPipeline(
    classifier_path=classifier_path,
    segmentation_path=segmentation_path,
    lesion_classifier_path=lesion_classifier_path,
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_FILE_SIZE_MB = 10

SEX_OPTIONS = {"female", "male", "unknown"}
LOCALIZATION_OPTIONS = {
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "unknown",
    "upper extremity",
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_meta_vector(sex: str, localization: str, age: int) -> np.ndarray:
    meta = {name: 0.0 for name in META_FEATURES_ORDER}

    sex_key = f"sex_{sex}"
    if sex_key not in meta:
        raise ValueError("Giới tính không hợp lệ")
    meta[sex_key] = 1.0

    localization_key = f"localization_{localization}"
    if localization_key not in meta:
        raise ValueError("Vị trí tổn thương không hợp lệ")
    meta[localization_key] = 1.0

    meta["age"] = float(age)

    return np.array([meta[name] for name in META_FEATURES_ORDER], dtype=np.float32)


@app.route("/health", methods=["GET"])
def health():
    """Kiểm tra server còn sống không"""
    return jsonify({"status": "ok", "models_loaded": pipeline.is_ready()})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint chính: nhận ảnh, chạy qua pipeline 3 model.

    Form-data:
        image (file): file ảnh cần phân tích

    Response JSON:
        stage            : giai đoạn dừng lại ("not_skin" | "normal_skin" | "lesion_found")
        is_skin          : bool
        has_lesion       : bool
        lesion_class     : str | null
        confidence       : float | null
        top_predictions  : list[{class, confidence}] | null
        mask_base64      : str | null  (ảnh mask PNG, base64-encoded)
        message          : str
    """
    # --- Validate input ---
    if "image" not in request.files:
        return jsonify({"error": "Thiếu field 'image' trong form-data"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Không có file được chọn"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Định dạng không hỗ trợ. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    file.seek(0, os.SEEK_END)
    file_size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if file_size_mb > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File quá lớn. Tối đa {MAX_FILE_SIZE_MB}MB"}), 400

    sex = (request.form.get("sex") or "").strip().lower()
    localization = (request.form.get("localization") or "").strip().lower()
    age_raw = (request.form.get("age") or "").strip()

    if sex not in SEX_OPTIONS:
        return jsonify({"error": "Giới tính không hợp lệ"}), 400

    if localization not in LOCALIZATION_OPTIONS:
        return jsonify({"error": "Vị trí tổn thương không hợp lệ"}), 400

    try:
        age = int(age_raw)
    except ValueError:
        return jsonify({"error": "Tuổi phải là số nguyên"}), 400

    if age < 1 or age > 100:
        return jsonify({"error": "Tuổi phải nằm trong khoảng 1-100"}), 400

    # --- Chạy pipeline ---
    try:
        image_bytes = file.read()
        meta_vector = build_meta_vector(sex, localization, age)
        result = pipeline.run(image_bytes, meta_vector=meta_vector)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": f"Lỗi dữ liệu: {str(e)}"}), 422
    except Exception as e:
        app.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        return jsonify({"error": "Lỗi server nội bộ"}), 500

@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    try:
        answer = answer_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "File quá lớn"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint không tồn tại"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method không được phép"}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
