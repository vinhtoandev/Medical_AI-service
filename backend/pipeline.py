import numpy as np
import cv2
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input



# ──────────────────────────────────────────────
#  Cấu hình các nhãn – chỉnh theo model của bạn
# ──────────────────────────────────────────────
LESION_CLASSES = [
    "dày sừng ánh sáng (actinic keratosis)",
    "Ung thư biểu mô tế bào đáy (BCC)",
    "tổn thương dày sừng lành tính (BKL)",
    "U sợi bì (Dermatofibroma)",
    "Ung thư hắc tố da (Melanoma)",
    "U hắc tố lành tính  (nevus)",
    "tổn thương mạch máu (Vascular lesions)"
]

# Metadata feature order must match training
META_FEATURES_ORDER = [
    "sex_female",
    "sex_male",
    "sex_unknown",
    "localization_abdomen",
    "localization_acral",
    "localization_back",
    "localization_chest",
    "localization_ear",
    "localization_face",
    "localization_foot",
    "localization_genital",
    "localization_hand",
    "localization_lower extremity",
    "localization_neck",
    "localization_scalp",
    "localization_trunk",
    "localization_unknown",
    "localization_upper extremity",
    "age",
]

# Ngưỡng quyết định
SKIN_THRESHOLD        = 0.5   # Model 1: confidence > ngưỡng → là da
MASK_AREA_THRESHOLD   = 0.01  # Model 2: lesion phải chiếm > 1% diện tích ảnh
TOP_K_PREDICTIONS     = 3     # Model 3: trả về top-3


class SkinAnalysisPipeline:
    def __init__(self, classifier_path, segmentation_path, lesion_classifier_path):
        print("[Pipeline] Đang load models...")
        
        self.skin_classifier   = tf.keras.models.load_model(classifier_path, safe_mode=False)
        self.segmentation      = tf.keras.models.load_model(segmentation_path, safe_mode=False)
        self.lesion_classifier = tf.keras.models.load_model(lesion_classifier_path, safe_mode=False)
        print("[Pipeline] Load xong tất cả models.")



    def is_ready(self):
        return all([
            self.skin_classifier is not None,
            self.segmentation is not None,
            self.lesion_classifier is not None
        ])

    # ── Utilities ──────────────────────────────

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        try:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            raise ValueError("Không thể đọc file ảnh. Hãy kiểm tra lại định dạng.")

    def _resize(self, pil_img: Image.Image, size: tuple) -> np.ndarray:
        """Resize và normalize về [0, 1]"""
        img = pil_img.resize(size, Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _add_batch(self, arr: np.ndarray) -> np.ndarray:
        return np.expand_dims(arr, axis=0)

    def _mask_to_base64(self, mask: np.ndarray) -> str:
        """Chuyển mask (H, W) float [0,1] → ảnh PNG base64"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        _, buffer = cv2.imencode(".png", mask_uint8)
        return base64.b64encode(buffer).decode("utf-8")

    def _overlay_to_base64(self, img, mask, alpha=0.4):

        img_uint8 = (img * 255).astype(np.uint8)

        # squeeze mask
        mask = np.squeeze(mask)

        mask_bin = mask > 0.5

        overlay = img_uint8.copy()

        # tạo màu đỏ
        color = np.zeros_like(img_uint8)
        color[:, :, 2] = 255

        overlay[mask_bin] = (
            (1 - alpha) * img_uint8[mask_bin]
            + alpha * color[mask_bin]
        ).astype(np.uint8)

        _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return base64.b64encode(buffer).decode("utf-8")

    def _apply_mask_to_image(self, img_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Crop vùng lesion theo bounding box của mask trên ảnh 256x256"""
        mask_bin = (mask > 0.5).astype(np.uint8)
        ys, xs = np.where(mask_bin == 1)

        if len(xs) == 0 or len(ys) == 0:
            return img_array

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(256, x_max + padding)
        y_max = min(256, y_max + padding)

        return img_array[y_min:y_max, x_min:x_max]

    # ── 3 bước pipeline ────────────────────────

    def _step1_is_skin(self, pil_img: Image.Image):
        """
        Model 1: Binary classification skin / not skin
        Input size: (224, 224) — chỉnh nếu model của bạn khác
        Output: (1, 1) sigmoid hoặc (1, 2) softmax
        """
        img = self._resize(pil_img, (224, 224))
        pred = self.skin_classifier.predict(self._add_batch(img), verbose=0)

        if pred.shape[-1] == 1:          # sigmoid output
            confidence = float(pred[0][0])
            is_skin = confidence > SKIN_THRESHOLD
        else:                             # softmax output  [not_skin, skin]
            confidence = float(pred[0][1])
            is_skin = confidence > SKIN_THRESHOLD

        return is_skin, confidence

    def _step2_segment(self, pil_img: Image.Image):
        """
        Model 2: Segmentation – trả về mask (H, W) float [0,1]
        Input size: (256, 256) — chỉnh nếu model của bạn khác
        """
        img = self._resize(pil_img, (256, 256))
        pred = self.segmentation.predict(self._add_batch(img), verbose=0)

        mask = pred[0, :, :, 0]  # shape (256, 256)

        # Tính tỉ lệ vùng lesion
        binary_mask = (mask > 0.5).astype(np.float32)
        lesion_ratio = float(binary_mask.sum()) / binary_mask.size
        has_lesion = lesion_ratio > MASK_AREA_THRESHOLD

        return has_lesion, mask, lesion_ratio, img

    def _step3_classify_lesion(self, cropped_array: np.ndarray):
        """
        Model 3: Multi-class lesion classification
        Input size: (224, 224) — chỉnh nếu model của bạn khác
        """
        cropped_img = Image.fromarray((cropped_array * 255).astype(np.uint8))
        cropped_img = cropped_img.resize((224, 224))
        img_array = np.array(cropped_img, dtype=np.float32)
        img_array = preprocess_input(img_array)
        pred = self.lesion_classifier.predict(self._add_batch(img_array), verbose=0)[0]

        # Top-K predictions
        top_indices = pred.argsort()[::-1][:TOP_K_PREDICTIONS]
        top_predictions = [
            {
                "class": LESION_CLASSES[i] if i < len(LESION_CLASSES) else f"Class_{i}",
                "confidence": round(float(pred[i]), 4),
            }
            for i in top_indices
        ]

        best_idx = int(pred.argmax())
        best_class = LESION_CLASSES[best_idx] if best_idx < len(LESION_CLASSES) else f"Class_{best_idx}"
        best_conf  = round(float(pred[best_idx]), 4)

        return best_class, best_conf, top_predictions

    def _step3_classify_lesion_with_meta(self, cropped_array: np.ndarray, meta_vector: np.ndarray):
        """
        Model 3: Multi-class lesion classification (multimodal)
        Inputs:
          - image_input: (224, 224, 3)
          - meta_input : (19,)
        """
        cropped_img = Image.fromarray((cropped_array * 255).astype(np.uint8))
        cropped_img = cropped_img.resize((224, 224))
        img_array = np.array(cropped_img, dtype=np.float32)
        img_array = preprocess_input(img_array)

        image_batch = self._add_batch(img_array)
        meta_batch = self._add_batch(meta_vector.astype(np.float32))

        pred = self.lesion_classifier.predict(
            {"image_input": image_batch, "meta_input": meta_batch},
            verbose=0,
        )[0]

        top_indices = pred.argsort()[::-1][:TOP_K_PREDICTIONS]
        top_predictions = [
            {
                "class": LESION_CLASSES[i] if i < len(LESION_CLASSES) else f"Class_{i}",
                "confidence": round(float(pred[i]), 4),
            }
            for i in top_indices
        ]

        best_idx = int(pred.argmax())
        best_class = LESION_CLASSES[best_idx] if best_idx < len(LESION_CLASSES) else f"Class_{best_idx}"
        best_conf = round(float(pred[best_idx]), 4)

        return best_class, best_conf, top_predictions

    # ── Entry point ────────────────────────────

    def run(self, image_bytes: bytes, meta_vector: np.ndarray | None = None) -> dict:
        pil_img = self._bytes_to_pil(image_bytes)

        # ── Bước 1: Có phải ảnh da không? ──
        is_skin, skin_confidence = self._step1_is_skin(pil_img)

        if not is_skin:
            return {
                "stage":           "not_skin",
                "is_skin":         False,
                "skin_confidence": round(skin_confidence, 4),
                "has_lesion":      False,
                "lesion_class":    None,
                "confidence":      None,
                "top_predictions": None,
                "mask_base64":     None,
                "message":         "Ảnh không chứa vùng da. Pipeline dừng ở bước 1.",
            }

        # ── Bước 2: Segmentation ──
        has_lesion, mask, lesion_ratio, resized_img = self._step2_segment(pil_img)
        mask_base64 = self._mask_to_base64(mask)
        overlay_base64 = self._overlay_to_base64(resized_img, mask)

        if not has_lesion:
            return {
                "stage":           "normal_skin",
                "is_skin":         True,
                "skin_confidence": round(skin_confidence, 4),
                "has_lesion":      False,
                "lesion_ratio":    round(lesion_ratio, 4),
                "lesion_class":    None,
                "confidence":      None,
                "top_predictions": None,
                "mask_base64":     mask_base64,
                "overlay_base64":  overlay_base64,
                "message":         "Da bình thường. Không phát hiện tổn thương.",
            }

        # ── Bước 3: Classify lesion ──
        # Crop vùng lesion trước khi đưa vào model 3
        cropped_array = self._apply_mask_to_image(resized_img, mask)
        if meta_vector is None:
            lesion_class, confidence, top_predictions = self._step3_classify_lesion(cropped_array)
        else:
            lesion_class, confidence, top_predictions = self._step3_classify_lesion_with_meta(
                cropped_array,
                meta_vector,
            )

        return {
            "stage":           "lesion_found",
            "is_skin":         True,
            "skin_confidence": round(skin_confidence, 4),
            "has_lesion":      True,
            "lesion_ratio":    round(lesion_ratio, 4),
            "lesion_class":    lesion_class,
            "confidence":      confidence,
            "top_predictions": top_predictions,
            "mask_base64":     mask_base64,
            "overlay_base64":  overlay_base64,
            "message":         f"Phát hiện tổn thương: {lesion_class} ({confidence*100:.1f}%)",
        }
