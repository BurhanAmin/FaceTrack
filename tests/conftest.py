import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database import DatabaseManager  # noqa: E402
from services.attendance_service import AttendanceService  # noqa: E402
from services.report_service import ReportService  # noqa: E402
from services.student_service import StudentService  # noqa: E402
from vision.face_recognizer import FaceRecognizer  # noqa: E402


@pytest.fixture
def db(tmp_path):
    return DatabaseManager(tmp_path / "test.db")


@pytest.fixture
def recognizer(tmp_path):
    return FaceRecognizer(tmp_path / "m.yml", tmp_path / "labels.json", tmp_path / "samples")


@pytest.fixture
def students(db, recognizer):
    return StudentService(db, recognizer)


@pytest.fixture
def attendance(db):
    return AttendanceService(db)


@pytest.fixture
def reports(db):
    return ReportService(db)
