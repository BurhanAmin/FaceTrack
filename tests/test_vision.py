import numpy as np

import config
from vision.face_detector import DetectionParams, FaceDetector, crop_face
from vision.image_processor import ImageLoadError, load_image_bytes, run_lab
import pytest


def test_detector_loads_and_handles_blank_frame():
    det = FaceDetector()
    boxes, gray = det.detect(np.zeros((240, 320, 3), np.uint8), DetectionParams(min_size=40))
    assert boxes == [] and gray.ndim == 2


def test_missing_cascade_raises(tmp_path):
    from vision.face_detector import DetectorError
    with pytest.raises(DetectorError):
        FaceDetector(tmp_path / "nope.xml")


def test_crop_face_size():
    g = (np.random.rand(300, 300) * 255).astype("uint8")
    assert crop_face(g, (10, 10, 100, 100)).shape[::-1] == config.FACE_SIZE


def test_invalid_upload_rejected():
    with pytest.raises(ImageLoadError):
        load_image_bytes(b"not an image")


def test_lab_runs():
    img = (np.random.rand(200, 200, 3) * 255).astype("uint8")
    r = run_lab(FaceDetector(), img, 1.2, 5, 30, 5)
    assert r.plain.count >= 0 and r.gray.shape == (200, 200)


def test_recognizer_distinguishes_two_synthetic_identities(recognizer):
    rng = np.random.default_rng(1)
    base = {"A": rng.random((200, 200)), "B": rng.random((200, 200))}
    for sid, b in base.items():
        for _ in range(8):
            recognizer.save_sample(sid, ((b * 0.9 + rng.random((200, 200)) * 0.1) * 255).astype("uint8"))
    recognizer.train(["A", "B"])
    probe = ((base["B"] * 0.9 + rng.random((200, 200)) * 0.1) * 255).astype("uint8")
    assert recognizer.predict(probe, 500).student_id == "B"
    assert recognizer.predict(probe, 0.001).student_id is None   # threshold -> Unknown
    recognizer.load()
    assert recognizer.is_ready
