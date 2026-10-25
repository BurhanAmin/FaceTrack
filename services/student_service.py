"""Student registration, search, deactivation and deletion."""
from __future__ import annotations

import logging
import re
from datetime import datetime

import pandas as pd

from database import DatabaseManager
from vision.face_recognizer import FaceRecognizer, RecognizerError

log = logging.getLogger("facetrack.students")

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class StudentError(Exception):
    """Validation / business-rule failure, safe to show to users."""


class StudentService:
    def __init__(self, db: DatabaseManager, recognizer: FaceRecognizer):
        self.db, self.recognizer = db, recognizer

    # ---- validation / create ----------------------------------------
    @staticmethod
    def validate(student_id: str, name: str, department: str, class_year: str, email: str = "") -> list[str]:
        errors = []
        if not student_id.strip():
            errors.append("Student ID is required.")
        elif not ID_RE.match(student_id.strip()):
            errors.append("Student ID may only contain letters, digits, '-' and '_' (max 32 chars).")
        if not name.strip():
            errors.append("Full name is required.")
        if not department.strip():
            errors.append("Course / department is required.")
        if not class_year.strip():
            errors.append("Year / class is required.")
        if email.strip() and not EMAIL_RE.match(email.strip()):
            errors.append("Email address looks invalid.")
        return errors

    def exists(self, student_id: str) -> bool:
        return self.db.query_one("SELECT 1 FROM students WHERE student_id=?", (student_id.strip(),)) is not None

    def add_student(self, student_id, name, department, class_year, email="") -> None:
        errors = self.validate(student_id, name, department, class_year, email)
        if errors:
            raise StudentError(" ".join(errors))
        if self.exists(student_id):
            raise StudentError(f"Student ID '{student_id.strip()}' is already registered.")
        self.db.execute(
            "INSERT INTO students (student_id, full_name, department, class_year, email, "
            "registration_date, active_status) VALUES (?,?,?,?,?,?,1)",
            (student_id.strip(), name.strip(), department.strip(), class_year.strip(),
             email.strip() or None, datetime.now().strftime("%Y-%m-%d %H:%M")))

    def remove_student_row_only(self, student_id: str) -> None:
        """Roll back a registration that produced no usable samples."""
        self.db.execute("DELETE FROM students WHERE student_id=?", (student_id,))

    # ---- queries -----------------------------------------------------
    def list_students(self, search: str = "", active_only: bool = False) -> pd.DataFrame:
        sql = ("SELECT student_id, full_name, department, class_year, email, registration_date, "
               "active_status FROM students WHERE 1=1")
        params: list = []
        if search.strip():
            sql += " AND (student_id LIKE ? OR full_name LIKE ?)"
            like = f"%{search.strip()}%"
            params += [like, like]
        if active_only:
            sql += " AND active_status=1"
        sql += " ORDER BY full_name COLLATE NOCASE"
        df = pd.DataFrame([dict(r) for r in self.db.query(sql, params)],
                          columns=["student_id", "full_name", "department", "class_year", "email",
                                   "registration_date", "active_status"])
        return df

    def get(self, student_id: str):
        return self.db.query_one("SELECT * FROM students WHERE student_id=?", (student_id,))

    def active_ids(self) -> list[str]:
        return [r["student_id"] for r in self.db.query("SELECT student_id FROM students WHERE active_status=1")]

    # ---- model -------------------------------------------------------
    def retrain(self) -> dict:
        """Retrain LBPH from active students' samples."""
        return self.recognizer.train(self.active_ids())

    def retrain_safely(self) -> str:
        try:
            r = self.retrain()
            return f"Model retrained on {r['images']} images from {r['students']} student(s)."
        except RecognizerError as exc:
            return str(exc)

    # ---- lifecycle ---------------------------------------------------
    def set_active(self, student_id: str, active: bool) -> None:
        if not self.exists(student_id):
            raise StudentError("Student not found.")
        self.db.execute("UPDATE students SET active_status=? WHERE student_id=?", (1 if active else 0, student_id))
        self.retrain_safely()

    def delete_samples_only(self, student_id: str) -> int:
        n = self.recognizer.delete_samples(student_id)
        self.retrain_safely()
        return n

    def delete_student(self, student_id: str) -> dict:
        """Delete the student, their attendance rows and their own sample folder only."""
        if not self.exists(student_id):
            raise StudentError("Student not found.")
        with self.db.connect() as conn:
            att = conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,)).rowcount
            conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))
        files = self.recognizer.delete_samples(student_id)
        # Free the label so a stale mapping can never match a removed student.
        self.recognizer.label_map = {k: v for k, v in self.recognizer.label_map.items() if v != student_id}
        msg = self.retrain_safely()
        return {"attendance_rows": att, "sample_files": files, "model": msg}
