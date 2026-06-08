# web/tests/test_pipeline.py
import numpy as np
from PIL import Image

from web.pipeline import SkinAnalysisPipeline, SKIN_THRESHOLD


class DummyClassifier:
    def __init__(self, pred):
        self._pred = pred

    def predict(self, x, verbose=0):
        return self._pred


def _make_pipeline_with_pred(pred):
    pipeline = SkinAnalysisPipeline.__new__(SkinAnalysisPipeline)  # bypass __init__
    pipeline.skin_classifier = DummyClassifier(pred)
    return pipeline


def _dummy_image():
    return Image.new("RGB", (300, 300), color=(128, 128, 128))


def test_step1_sigmoid_true():
    pred = np.array([[SKIN_THRESHOLD + 0.1]], dtype=np.float32)  # shape (1,1)
    pipeline = _make_pipeline_with_pred(pred)

    is_skin, conf = pipeline._step1_is_skin(_dummy_image())

    assert is_skin is True
    assert abs(conf - float(pred[0][0])) < 1e-6


def test_step1_sigmoid_false():
    pred = np.array([[SKIN_THRESHOLD - 0.1]], dtype=np.float32)
    pipeline = _make_pipeline_with_pred(pred)

    is_skin, conf = pipeline._step1_is_skin(_dummy_image())

    assert is_skin is False
    assert abs(conf - float(pred[0][0])) < 1e-6


def test_step1_softmax_true():
    pred = np.array([[0.2, SKIN_THRESHOLD + 0.05]], dtype=np.float32)  # shape (1,2)
    pipeline = _make_pipeline_with_pred(pred)

    is_skin, conf = pipeline._step1_is_skin(_dummy_image())

    assert is_skin is True
    assert abs(conf - float(pred[0][1])) < 1e-6


def test_step1_softmax_false():
    pred = np.array([[0.9, SKIN_THRESHOLD - 0.05]], dtype=np.float32)
    pipeline = _make_pipeline_with_pred(pred)

    is_skin, conf = pipeline._step1_is_skin(_dummy_image())

    assert is_skin is False
    assert abs(conf - float(pred[0][1])) < 1e-6