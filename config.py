"""Central configuration for FaceTrack."""
from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# FACETRACK_DATA_DIR lets tests / demos use a separate data folder.
DATA_DIR = Path(os.environ.get("FACETRACK_DATA_DIR", BASE_DIR / "data"))
FACE_SAMPLES_DIR = DATA_DIR / "face_samples"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = DATA_DIR / "facetrack.db"
LOG_PATH = DATA_DIR / "facetrack.log"
CSS_PATH = BASE_DIR / "assets" / "styles.css"

MODEL_PATH = MODELS_DIR / "lbph_model.yml"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

APP_TITLE = "FaceTrack — Smart Attendance System"
APP_SUBTITLE = "AI-assisted attendance powered by OpenCV"

# Detection defaults (user-adjustable in the UI)
DEFAULT_SCALE_FACTOR = 1.2
DEFAULT_MIN_NEIGHBORS = 5
DEFAULT_MIN_FACE_SIZE = 80

# Every face crop is resized to this size before training / prediction.
FACE_SIZE = (200, 200)

# LBPH distance threshold: LOWER distance = better match. It is NOT a probability.
DEFAULT_RECOGNITION_THRESHOLD = 70.0
DEFAULT_CONSECUTIVE_FRAMES = 5

SAMPLES_PER_STUDENT = 20
MIN_SAMPLES_TO_TRAIN = 5


def ensure_dirs() -> None:
    for d in (DATA_DIR, FACE_SAMPLES_DIR, MODELS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    root = logging.getLogger("facetrack")
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
