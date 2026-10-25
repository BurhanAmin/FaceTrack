"""Webcam helper that always releases the device."""
from __future__ import annotations

import logging
import time

import cv2

log = logging.getLogger("facetrack.camera")


class CameraError(Exception):
    """Webcam missing, busy, or permission denied."""


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480):
        self.index, self.width, self.height = index, width, height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                "Could not open the webcam. Check that it is connected, not used by another app, "
                "and that camera permission is granted to your terminal / Python "
                "(macOS: System Settings > Privacy & Security > Camera).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        # Some cameras (esp. macOS) return black/empty frames while warming up.
        for _ in range(10):
            ok, frame = cap.read()
            if ok and frame is not None:
                return self
            time.sleep(0.1)
        self.release()
        raise CameraError("The webcam opened but delivered no frames. Check camera permission.")

    def read(self):
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()
