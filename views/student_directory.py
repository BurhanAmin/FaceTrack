import streamlit as st

from services.student_service import StudentError
from vision.face_recognizer import RecognizerError
from views._ui import chip, page_header, section


def render(svc) -> None:
    page_header("Student Directory", "Search, review, deactivate or remove registered students")
    search = st.text_input("Search by student ID or name", placeholder="e.g. 2024-017 or Priya")
    df = svc.students.list_students(search)
    if df.empty:
        st.info("No students found." if search else "No students are registered yet.")
        return

    show = df.assign(status=df["active_status"].map({1: "Active", 0: "Deactivated"}),
                     samples=df["student_id"].map(svc.recognizer.sample_count)).drop(columns="active_status")
    show.columns = ["Student ID", "Name", "Department", "Class", "Email", "Registered", "Status", "Face samples"]
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(f"{len(df)} student(s)")

    section("Student details")
    label = {r.student_id: f"{r.student_id} — {r.full_name}" for r in df.itertuples()}
    sid = st.selectbox("Select a student", list(label), format_func=label.get)
    stu = svc.students.get(sid)
    active = bool(stu["active_status"])
    st.markdown(f"**{stu['full_name']}** · {stu['department']} · {stu['class_year']} · "
                + chip("Active" if active else "Deactivated", "ok" if active else "warn"),
                unsafe_allow_html=True)

    hist, stats = svc.reports.student_report(sid)
    c = st.columns(3)
    c[0].metric("Days present", stats["present"])
    c[1].metric("Session days since registration", stats["sessions"])
    c[2].metric("Attendance", f"{stats['percent']}%")
    if hist.empty:
        st.caption("No attendance history yet.")
    else:
        st.dataframe(hist, hide_index=True, width="stretch")

    section("Actions")
    a, b = st.columns(2)
    try:
        if active:
            if a.button("Deactivate student", help="Excluded from recognition and absent counts; history kept."):
                svc.students.set_active(sid, False)
                st.rerun()
        else:
            if a.button("Reactivate student"):
                svc.students.set_active(sid, True)
                st.rerun()
        if b.button("Delete face samples only", help="Removes this student's face images and retrains."):
            n = svc.students.delete_samples_only(sid)
            st.success(f"Deleted {n} face sample file(s). The student can no longer be recognized until re-registered "
                       "with new samples.")
    except (StudentError, RecognizerError, OSError) as exc:
        st.error(str(exc))

    with st.expander("Delete this student permanently"):
        st.warning("This removes the student record, ALL of their attendance records and their face samples. "
                   "It cannot be undone. Other students are not affected.")
        confirm = st.text_input(f"Type the student ID ({sid}) to confirm", key=f"del_{sid}")
        if st.button("Delete permanently", type="primary", disabled=confirm.strip() != sid):
            try:
                r = svc.students.delete_student(sid)
                st.success(f"Deleted {sid}: {r['attendance_rows']} attendance row(s) and "
                           f"{r['sample_files']} sample file(s) removed. {r['model']}")
            except (StudentError, RecognizerError, OSError) as exc:
                st.error(str(exc))

    section("Recognition model")
    if st.button("Retrain model now"):
        st.info(svc.students.retrain_safely())
