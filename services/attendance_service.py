"""Attendance rules: one record per student per day, no duplicate writes."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from database import DatabaseManager

log = logging.getLogger("facetrack.attendance")

MARKED, ALREADY, REJECTED = "marked", "already_marked", "rejected"


@dataclass
class MarkResult:
    status: str
    message: str


class AttendanceService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def mark(self, student_id: str, distance: float | None = None, when: datetime | None = None) -> MarkResult:
        when = when or datetime.now()
        student = self.db.query_one("SELECT active_status FROM students WHERE student_id=?", (student_id,))
        if student is None or not student["active_status"]:
            return MarkResult(REJECTED, "Student is not registered or is deactivated.")
        try:
            self.db.execute(
                "INSERT INTO attendance (student_id, attendance_date, attendance_time, recognition_distance) "
                "VALUES (?,?,?,?)",
                (student_id, when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S"),
                 None if distance is None else round(float(distance), 2)))
        except sqlite3.IntegrityError:
            return MarkResult(ALREADY, "Already marked")
        return MarkResult(MARKED, "Attendance marked")

    def is_marked(self, student_id: str, date: str | None = None) -> bool:
        date = date or datetime.now().strftime("%Y-%m-%d")
        return self.db.query_one(
            "SELECT 1 FROM attendance WHERE student_id=? AND attendance_date=?", (student_id, date)) is not None

    def recent(self, limit: int = 10) -> pd.DataFrame:
        rows = self.db.query(
            "SELECT a.attendance_date AS date, a.attendance_time AS time, s.student_id, s.full_name AS name, "
            "a.recognition_distance AS distance FROM attendance a JOIN students s USING(student_id) "
            "ORDER BY a.attendance_id DESC LIMIT ?", (limit,))
        return pd.DataFrame([dict(r) for r in rows], columns=["date", "time", "student_id", "name", "distance"])

    def count_on(self, date: str) -> int:
        return self.db.query_one("SELECT COUNT(*) c FROM attendance WHERE attendance_date=?", (date,))["c"]


@dataclass
class RecognitionTracker:
    """Marks a student only after N consecutive frames recognise the same ID.

    Also remembers who has been handled this session so the database is not
    written again for the same student.
    """
    required: int = 5
    handled: set[str] = field(default_factory=set)
    _counts: dict[str, int] = field(default_factory=dict)

    def update(self, recognized_ids: list[str | None]) -> list[str]:
        """Feed the IDs seen in one frame (None = unknown). Returns IDs ready to mark."""
        seen = {i for i in recognized_ids if i}
        # A student's streak resets as soon as they are missed or misread in a frame.
        self._counts = {sid: self._counts.get(sid, 0) + 1 for sid in seen}
        ready = []
        for sid, n in self._counts.items():
            if n >= self.required and sid not in self.handled:
                self.handled.add(sid)
                ready.append(sid)
        return ready
