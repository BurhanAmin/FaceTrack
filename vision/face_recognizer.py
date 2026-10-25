"""Face RECOGNITION (whose face is it?) using OpenCV's LBPH recognizer.

LBPH (Local Binary Patterns Histograms) compares a new face to the stored
training faces and returns the closest label plus a DISTANCE. Lower distance =
more similar. The distance is NOT a probability or percentage.

Labels: LBPH needs integer labels. We keep a persistent label_map.json
({label: student_id}) where labels are never reused, so the mapping does not
depend on the order students are loaded.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger("facetrack.recognizer")

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


class RecognizerError(Exception):
    """User-presentable recognition / training failure."""


@dataclass
class Prediction:
    student_id: str | None   # None => Unknown
    distance: float | None


def student_sample_dir(student_id: str, base: Path | None = None) -> Path:
    return (base or config.FACE_SAMPLES_DIR) / student_id


def _safe_id(student_id: str) -> str:
    if not student_id or any(c in student_id for c in "/\\.") or student_id.strip() != student_id:
        raise RecognizerError("Invalid student ID for file storage.")
    return student_id


class FaceRecognizer:
    def __init__(self, model_path: Path | None = None, label_map_path: Path | None = None,
                 samples_dir: Path | None = None):
        self.model_path = model_path or config.MODEL_PATH
        self.label_map_path = label_map_path or config.LABEL_MAP_PATH
        self.samples_dir = samples_dir or config.FACE_SAMPLES_DIR
        self._model = None
        self.label_map: dict[int, str] = {}
        self._load_label_map()

    # ---- persistence -------------------------------------------------
    def _load_label_map(self) -> None:
        try:
            if self.label_map_path.exists():
                raw = json.loads(self.label_map_path.read_text(encoding="utf-8"))
                self.label_map = {int(k): v for k, v in raw.items()}
        except (OSError, ValueError):
            log.exception("Label map unreadable; starting empty")
            self.label_map = {}

    def _save_label_map(self) -> None:
        self.label_map_path.parent.mkdir(parents=True, exist_ok=True)
        self.label_map_path.write_text(
            json.dumps({str(k): v for k, v in self.label_map.items()}, indent=2), encoding="utf-8")

    def _label_for(self, student_id: str) -> int:
        for label, sid in self.label_map.items():
            if sid == student_id:
                return label
        label = max(self.label_map, default=-1) + 1
        self.label_map[label] = student_id
        return label

    @staticmethod
    def _new_model():
        if not hasattr(cv2, "face"):
            raise RecognizerError(
                "cv2.face is missing. Install 'opencv-contrib-python' (and uninstall 'opencv-python').")
        return cv2.face.LBPHFaceRecognizer_create()

    def load(self) -> bool:
        """Load the saved model. Returns False if absent; raises if corrupted."""
        self._model = None
        if not self.model_path.exists():
            return False
        try:
            model = self._new_model()
            model.read(str(self.model_path))
            self._model = model
            return True
        except RecognizerError:
            raise
        except cv2.error as exc:
            log.exception("Model file corrupted")
            raise RecognizerError(
                "The saved recognition model is corrupted. Retrain it from Register Student "
                "or the Student Directory.") from exc

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    # ---- samples -----------------------------------------------------
    def save_sample(self, student_id: str, face_img: np.ndarray) -> Path:
        d = student_sample_dir(_safe_id(student_id), self.samples_dir)
        d.mkdir(parents=True, exist_ok=True)
        n = len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        path = d / f"{n + 1:03d}.png"
        while path.exists():
            n += 1
            path = d / f"{n + 1:03d}.png"
        if not cv2.imwrite(str(path), face_img):
            raise RecognizerError("Could not write the face sample (check folder permissions).")
        return path

    def sample_count(self, student_id: str) -> int:
        d = student_sample_dir(student_id, self.samples_dir)
        return len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]) if d.is_dir() else 0

    def delete_samples(self, student_id: str) -> int:
        """Delete only this student's sample files, then remove the folder if empty."""
        d = student_sample_dir(_safe_id(student_id), self.samples_dir)
        if not d.is_dir():
            return 0
        removed = 0
        for p in d.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                p.unlink()
                removed += 1
        try:
            d.rmdir()
        except OSError:
            log.warning("Sample folder %s not empty; left in place", d)
        return removed

    # ---- training ----------------------------------------------------
    def train(self, student_ids: list[str]) -> dict:
        """Train from sample folders of the given (active) students; save model + labels."""
        images, labels, used = [], [], set()
        for sid in student_ids:
            d = student_sample_dir(_safe_id(sid), self.samples_dir)
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    log.warning("Skipping unreadable sample %s", p)
                    continue
                if img.shape[::-1] != config.FACE_SIZE:
                    img = cv2.resize(img, config.FACE_SIZE)
                images.append(img)
                labels.append(self._label_for(sid))
                used.add(sid)
        if not images:
            self._model = None
            if self.model_path.exists():
                self.model_path.unlink()
            raise RecognizerError("No face samples available for training. Register a student first.")
        model = self._new_model()
        model.train(images, np.array(labels, dtype=np.int32))
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(self.model_path))
        self._save_label_map()
        self._model = model
        log.info("Trained LBPH on %d images for %d students", len(images), len(used))
        return {"images": len(images), "students": len(used)}

    # ---- prediction --------------------------------------------------
    def predict(self, face_img: np.ndarray, threshold: float) -> Prediction:
        """Return the matched student if distance <= threshold, else Unknown."""
        if self._model is None:
            return Prediction(None, None)
        label, distance = self._model.predict(face_img)
        sid = self.label_map.get(int(label))
        if sid is None or distance > threshold:
            return Prediction(None, float(distance))
        return Prediction(sid, float(distance))
