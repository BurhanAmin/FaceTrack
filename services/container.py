"""Builds the shared service objects once per Streamlit server process."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import config
from database import DatabaseManager
from services.attendance_service import AttendanceService
from services.report_service import ReportService
from services.student_service import StudentService
from vision.face_detector import DetectorError, FaceDetector
from vision.face_recognizer import FaceRecognizer, RecognizerError

log = logging.getLogger("facetrack.container")


@dataclass
class Services:
    db: DatabaseManager
    detector: FaceDetector | None
    detector_error: str | None
    recognizer: FaceRecognizer
    model_error: str | None
    students: StudentService
    attendance: AttendanceService
    reports: ReportService


def build_services(db_path=None) -> Services:
    config.setup_logging()
    db = DatabaseManager(db_path)
    try:
        detector, det_err = FaceDetector(), None
    except DetectorError as exc:
        log.error("Detector unavailable: %s", exc)
        detector, det_err = None, str(exc)
    recognizer = FaceRecognizer()
    model_err = None
    try:
        recognizer.load()
    except RecognizerError as exc:
        model_err = str(exc)
    return Services(db, detector, det_err, recognizer, model_err,
                    StudentService(db, recognizer), AttendanceService(db), ReportService(db))
