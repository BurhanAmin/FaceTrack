"""Attendance reports. Definition used everywhere:

* One attendance record per student per day.
* A student is "enrolled" on date D if registration date <= D (and they are active,
  or already have a record on D). Students registered after D are NOT counted absent.
* Individual percentage = days present / "session days" since the student's
  registration, where a session day is any date on which at least one student
  was marked present.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from database import DatabaseManager

COLS = ["date", "time", "student_id", "name", "department", "class_year", "distance"]


class ReportService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    # ---- helpers -----------------------------------------------------
    def _enrolled_on(self, d: str) -> pd.DataFrame:
        rows = self.db.query(
            "SELECT s.student_id, s.full_name, s.department, s.class_year FROM students s "
            "WHERE substr(s.registration_date,1,10) <= ? AND (s.active_status=1 OR EXISTS "
            "(SELECT 1 FROM attendance a WHERE a.student_id=s.student_id AND a.attendance_date=?)) "
            "ORDER BY s.full_name COLLATE NOCASE", (d, d))
        return pd.DataFrame([dict(r) for r in rows], columns=["student_id", "full_name", "department", "class_year"])

    def _present_on(self, d: str) -> set[str]:
        return {r["student_id"] for r in self.db.query(
            "SELECT student_id FROM attendance WHERE attendance_date=?", (d,))}

    # ---- dashboard ---------------------------------------------------
    def daily_summary(self, d: str) -> dict:
        enrolled = self._enrolled_on(d)
        present = self._present_on(d) & set(enrolled["student_id"])
        total = len(enrolled)
        return {"enrolled": total, "present": len(present), "absent": total - len(present),
                "percent": round(100 * len(present) / total, 1) if total else 0.0}

    def total_registered(self) -> int:
        return self.db.query_one("SELECT COUNT(*) c FROM students")["c"]

    def last_days(self, n: int = 7, today: date | None = None) -> pd.DataFrame:
        today = today or date.today()
        out = []
        for i in range(n - 1, -1, -1):
            d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
            s = self.daily_summary(d)
            out.append({"date": d, "present": s["present"], "absent": s["absent"]})
        return pd.DataFrame(out)

    # ---- records page ------------------------------------------------
    def records(self, start: str | None = None, end: str | None = None, query: str = "",
                department: str = "", class_year: str = "") -> pd.DataFrame:
        sql = ("SELECT a.attendance_date AS date, a.attendance_time AS time, s.student_id, "
               "s.full_name AS name, s.department, s.class_year, a.recognition_distance AS distance "
               "FROM attendance a JOIN students s USING(student_id) WHERE 1=1")
        p: list = []
        if start:
            sql += " AND a.attendance_date >= ?"; p.append(start)
        if end:
            sql += " AND a.attendance_date <= ?"; p.append(end)
        if query.strip():
            sql += " AND (s.student_id LIKE ? OR s.full_name LIKE ?)"; p += [f"%{query.strip()}%"] * 2
        if department:
            sql += " AND s.department = ?"; p.append(department)
        if class_year:
            sql += " AND s.class_year = ?"; p.append(class_year)
        sql += " ORDER BY a.attendance_date DESC, a.attendance_time DESC"
        return pd.DataFrame([dict(r) for r in self.db.query(sql, p)], columns=COLS)

    def departments(self) -> list[str]:
        return [r[0] for r in self.db.query("SELECT DISTINCT department FROM students ORDER BY 1")]

    def class_years(self) -> list[str]:
        return [r[0] for r in self.db.query("SELECT DISTINCT class_year FROM students ORDER BY 1")]

    def daily_report(self, d: str) -> pd.DataFrame:
        """Every enrolled student with Present/Absent for date d."""
        enrolled = self._enrolled_on(d)
        times = {r["student_id"]: r["attendance_time"] for r in self.db.query(
            "SELECT student_id, attendance_time FROM attendance WHERE attendance_date=?", (d,))}
        enrolled["status"] = enrolled["student_id"].map(lambda s: "Present" if s in times else "Absent")
        enrolled["time"] = enrolled["student_id"].map(lambda s: times.get(s, ""))
        return enrolled.rename(columns={"full_name": "name"})

    def student_report(self, student_id: str) -> tuple[pd.DataFrame, dict]:
        stu = self.db.query_one("SELECT registration_date FROM students WHERE student_id=?", (student_id,))
        if stu is None:
            return pd.DataFrame(columns=["date", "status", "time"]), {"present": 0, "sessions": 0, "percent": 0.0}
        reg = stu["registration_date"][:10]
        session_days = [r[0] for r in self.db.query(
            "SELECT DISTINCT attendance_date FROM attendance WHERE attendance_date >= ? ORDER BY 1", (reg,))]
        mine = {r["attendance_date"]: r["attendance_time"] for r in self.db.query(
            "SELECT attendance_date, attendance_time FROM attendance WHERE student_id=?", (student_id,))}
        days = sorted(set(session_days) | set(mine), reverse=True)
        df = pd.DataFrame([{"date": d, "status": "Present" if d in mine else "Absent",
                            "time": mine.get(d, "")} for d in days], columns=["date", "status", "time"])
        sessions = len(days)
        present = len(mine)
        return df, {"present": present, "sessions": sessions,
                    "percent": round(100 * present / sessions, 1) if sessions else 0.0}
