"""Image-processing pipeline used by the Image Processing Lab."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from .face_detector import DetectionParams, FaceDetector, draw_box


class ImageLoadError(Exception):
    pass


def load_image_bytes(data: bytes, max_side: int = 1280) -> np.ndarray:
    """Decode an upload into a BGR image; reject corrupt files; shrink huge ones."""
    try:
        import io
        pil = Image.open(io.BytesIO(data))
        pil.verify()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError("The file is not a valid image or is corrupted.") from exc
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


@dataclass
class DetectionResult:
    annotated_rgb: np.ndarray
    count: int
    ms: float


@dataclass
class LabResult:
    original_rgb: np.ndarray
    gray: np.ndarray
    equalized: np.ndarray
    blurred: np.ndarray
    plain: DetectionResult
    equalized_det: DetectionResult
    gray_hist: np.ndarray
    eq_hist: np.ndarray


def _detect_and_draw(detector: FaceDetector, bgr: np.ndarray, params: DetectionParams) -> DetectionResult:
    t0 = time.perf_counter()
    boxes, _ = detector.detect(bgr, params)
    ms = (time.perf_counter() - t0) * 1000
    out = bgr.copy()
    for b in boxes:
        draw_box(out, b, "face")
    return DetectionResult(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), len(boxes), ms)


def run_lab(detector: FaceDetector, bgr: np.ndarray, scale_factor: float, min_neighbors: int,
            min_size: int, blur_kernel: int, apply_equalization: bool = True) -> LabResult:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    base = dict(scale_factor=scale_factor, min_neighbors=min_neighbors, min_size=min_size)
    plain = _detect_and_draw(detector, bgr, DetectionParams(equalize=False, **base))
    # Stage 6: detection after histogram equalization (toggle switches it to the plain run).
    eq = _detect_and_draw(detector, bgr, DetectionParams(equalize=apply_equalization, **base))
    return LabResult(
        original_rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), gray=gray, equalized=equalized,
        blurred=blurred, plain=plain, equalized_det=eq,
        gray_hist=cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel(),
        eq_hist=cv2.calcHist([equalized], [0], None, [256], [0, 256]).ravel(),
    )
