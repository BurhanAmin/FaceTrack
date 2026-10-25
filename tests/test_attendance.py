from datetime import date, datetime, timedelta

from services.attendance_service import ALREADY, MARKED, REJECTED, RecognitionTracker


def test_mark_once_per_day(students, attendance):
    students.add_student("S1", "A", "CS", "TY")
    t = datetime(2025, 3, 3, 9, 0)
    assert attendance.mark("S1", 33.3, t).status == MARKED
    second = attendance.mark("S1", 30.0, t + timedelta(minutes=5))
    assert second.status == ALREADY and second.message == "Already marked"
    assert attendance.count_on("2025-03-03") == 1
    # a new day is a new record
    assert attendance.mark("S1", 30.0, t + timedelta(days=1)).status == MARKED


def test_unregistered_or_inactive_rejected(students, attendance):
    assert attendance.mark("ghost").status == REJECTED
    students.add_student("S1", "A", "CS", "TY")
    students.set_active("S1", False)
    assert attendance.mark("S1").status == REJECTED


def test_tracker_requires_consecutive_frames():
    t = RecognitionTracker(required=3)
    assert t.update(["S1"]) == [] and t.update(["S1"]) == []
    assert t.update([None]) == []          # miss resets streak
    assert t.update(["S1"]) == [] and t.update(["S1"]) == []
    assert t.update(["S1"]) == ["S1"]
    assert t.update(["S1"]) == []          # handled: no repeated DB writes


def test_unknown_never_marks():
    t = RecognitionTracker(required=1)
    assert t.update([None, None]) == []


def test_reports_and_enrollment(db, students, attendance, reports):
    students.add_student("S1", "A", "CS", "TY")
    # backdate S1 registration; S2 registers "later"
    db.execute("UPDATE students SET registration_date='2025-03-01 08:00' WHERE student_id='S1'")
    students.add_student("S2", "B", "CS", "TY")
    db.execute("UPDATE students SET registration_date='2025-03-05 08:00' WHERE student_id='S2'")
    attendance.mark("S1", 30, datetime(2025, 3, 3, 9))
    attendance.mark("S1", 30, datetime(2025, 3, 5, 9))
    attendance.mark("S2", 30, datetime(2025, 3, 5, 9))
    d3 = reports.daily_summary("2025-03-03")
    assert d3 == {"enrolled": 1, "present": 1, "absent": 0, "percent": 100.0}  # S2 not yet enrolled
    d4 = reports.daily_summary("2025-03-04")
    assert d4["enrolled"] == 1 and d4["absent"] == 1
    _, s1 = reports.student_report("S1")
    assert s1 == {"present": 2, "sessions": 2, "percent": 100.0}
    _, s2 = reports.student_report("S2")
    assert s2 == {"present": 1, "sessions": 1, "percent": 100.0}


def test_empty_database(reports):
    assert reports.daily_summary("2025-01-01")["percent"] == 0.0
    assert len(reports.last_days(7)) == 7
    assert reports.records().empty
    assert reports.daily_report("2025-01-01").empty
