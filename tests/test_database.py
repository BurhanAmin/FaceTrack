import sqlite3

import numpy as np
import pytest

from services.student_service import StudentError


def test_schema_created(db):
    names = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"students", "attendance"} <= names


def test_add_and_list_student(students):
    students.add_student("S1", "Asha Rao", "CS", "TY", "a@b.co")
    df = students.list_students()
    assert list(df["student_id"]) == ["S1"]
    assert len(students.list_students("asha")) == 1
    assert students.list_students("zzz").empty


def test_duplicate_student_rejected(students):
    students.add_student("S1", "A", "CS", "TY")
    with pytest.raises(StudentError):
        students.add_student("S1", "B", "CS", "TY")


@pytest.mark.parametrize("args", [("", "A", "CS", "TY"), ("S1", "", "CS", "TY"),
                                  ("S1", "A", "", "TY"), ("S1", "A", "CS", ""),
                                  ("S/1", "A", "CS", "TY"), ("S1", "A", "CS", "TY", "bad-email")])
def test_validation(students, args):
    with pytest.raises(StudentError):
        students.add_student(*args)


def test_foreign_key_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO attendance (student_id, attendance_date, attendance_time) "
                   "VALUES ('ghost','2025-01-01','09:00:00')")


def test_unique_constraint_in_schema(db, students):
    students.add_student("S1", "A", "CS", "TY")
    sql = "INSERT INTO attendance (student_id, attendance_date, attendance_time) VALUES ('S1','2025-01-01','09:00:00')"
    db.execute(sql)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(sql)


def test_delete_student_only_touches_own_data(students, recognizer, attendance):
    students.add_student("S1", "A", "CS", "TY")
    students.add_student("S2", "B", "CS", "TY")
    for sid in ("S1", "S2"):
        for _ in range(6):
            recognizer.save_sample(sid, (np.random.rand(200, 200) * 255).astype("uint8"))
    attendance.mark("S1", 40.0)
    attendance.mark("S2", 41.0)
    r = students.delete_student("S1")
    assert r["attendance_rows"] == 1 and r["sample_files"] == 6
    assert recognizer.sample_count("S1") == 0 and recognizer.sample_count("S2") == 6
    assert students.exists("S2") and not students.exists("S1")
    assert len(attendance.recent()) == 1
    assert set(recognizer.label_map.values()) == {"S2"}


def test_training_and_stable_labels(students, recognizer):
    students.add_student("B", "B", "CS", "TY")
    students.add_student("A", "A", "CS", "TY")
    rng = np.random.default_rng(0)
    for sid in ("B", "A"):
        for _ in range(6):
            recognizer.save_sample(sid, (rng.random((200, 200)) * 255).astype("uint8"))
    students.retrain()
    first = dict(recognizer.label_map)
    students.set_active("B", False)       # retrain without B
    students.set_active("B", True)        # and with B again
    assert {v: k for k, v in recognizer.label_map.items()} == {v: k for k, v in first.items()}


def test_train_without_samples_raises(recognizer):
    from vision.face_recognizer import RecognizerError
    with pytest.raises(RecognizerError):
        recognizer.train(["nobody"])
