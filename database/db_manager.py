"""SQLite access layer. Schema is created automatically on first connect."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import config

log = logging.getLogger("facetrack.db")


class DatabaseError(Exception):
    """User-presentable database failure."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id        TEXT PRIMARY KEY,
    full_name         TEXT NOT NULL,
    department        TEXT NOT NULL,
    class_year        TEXT NOT NULL,
    email             TEXT,
    registration_date TEXT NOT NULL,
    active_status     INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id           TEXT NOT NULL,
    attendance_date      TEXT NOT NULL,
    attendance_time      TEXT NOT NULL,
    recognition_distance REAL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    UNIQUE (student_id, attendance_date)
);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(attendance_date);
"""


class DatabaseManager:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else config.DB_PATH
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except (OSError, sqlite3.Error) as exc:
            log.exception("Database initialisation failed")
            raise DatabaseError(
                f"Could not open the database at {self.db_path}. Check folder permissions."
            ) from exc

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a write statement; returns rowcount."""
        with self.connect() as conn:
            return conn.execute(sql, params).rowcount

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None
