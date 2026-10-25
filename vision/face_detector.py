"""Face DETECTION (where are the faces?) using a Haar Cascade classifier.

Detection answers "is there a face here, and where?". It knows nothing about
WHO the face belongs to - that is the job of vision/face_recognizer.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger("facetrack.detector")

CASCADE_NAME = "haarcascade_frontalface_default.xml"


class DetectorError(Exception):
    """Raised when the cascade file cannot be found or loaded."""


@dataclass(frozen=True)
class DetectionParams:
    scale_factor: float = config.DEFAULT_SCALE_FACTOR
    min_neighbors: int = config.DEFAULT_MIN_NEIGHBORS
    min_size: int = config.DEFAULT_MIN_FACE_SIZE
    equalize: bool = True


def to_gray(frame: np.ndarray) -> np.ndarray:
    """BGR -> single-channel intensity image (Haar features use intensity only)."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def preprocess(gray: np.ndarray, equalize: bool = True) -> np.ndarray:
    """Optional global histogram equalization to normalise lighting/contrast."""
    return cv2.equalizeHist(gray) if equalize else gray


class FaceDetector:
    def __init__(self, cascade_path: str | Path | None = None):
        path = Path(cascade_path) if cascade_path else Path(cv2.data.haarcascades) / CASCADE_NAME
        if not path.exists():
            raise DetectorError(f"Haar Cascade file not found: {path}")
        self._cascade = cv2.CascadeClassifier(str(path))
        if self._cascade.empty():
            raise DetectorError(f"Haar Cascade file could not be loaded: {path}")
        self.path = path

    def detect(self, frame: np.ndarray, params: DetectionParams | None = None):
        """Return (boxes, gray). boxes is a list of (x, y, w, h); empty list if none."""
        params = params or DetectionParams()
        gray = to_gray(frame)
        proc = preprocess(gray, params.equalize)
        faces = self._cascade.detectMultiScale(
            proc,
            scaleFactor=max(1.01, params.scale_factor),
            minNeighbors=max(0, params.min_neighbors),
            minSize=(params.min_size, params.min_size),
        )
        boxes = [tuple(int(v) for v in f) for f in faces] if len(faces) else []
        return boxes, gray


def crop_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop, equalize and resize a face so training and prediction see identical input."""
    x, y, w, h = box
    face = gray[max(0, y): y + h, max(0, x): x + w]
    face = cv2.resize(face, config.FACE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face)


def draw_box(frame, box, label: str, color=(80, 180, 40)) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(0, y - th - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 8, top + th + 8), color, -1)
    cv2.putText(frame, label, (x + 4, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
