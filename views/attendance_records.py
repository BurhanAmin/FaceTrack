from datetime import date

import streamlit as st

import config
from views._ui import note, page_header, section


def _save_copy(df, name: str) -> None:
    try:
        config.ensure_dirs()
        path = config.EXPORTS_DIR / name
        df.to_csv(path, index=False)
        st.success(f"Saved a copy to {path}")
    except OSError as exc:
        st.error(f"Could not save the file: {exc}")


def render(svc) -> None:
    page_header("Attendance Records", "Filter, export and report on attendance")
    note("Definition: one record per student per day. Students registered after a date are treated as "
         "not yet enrolled for that date, not absent.")
    t1, t2, t3 = st.tabs(["Records", "Daily report", "Student report"])

    with t1:
        c = st.columns(4)
        use_range = c[0].checkbox("Filter by date", value=False)
        d_start = c[0].date_input("From", date.today(), disabled=not use_range)
        d_end = c[1].date_input("To", date.today(), disabled=not use_range)
        query = c[1 + 1].text_input("Student ID or name")
        dept = c[3].selectbox("Department", [""] + svc.reports.departments(), format_func=lambda x: x or "All")
        cls = c[3].selectbox("Class / year", [""] + svc.reports.class_years(), format_func=lambda x: x or "All")
        df = svc.reports.records(d_start.isoformat() if use_range else None,
                                 d_end.isoformat() if use_range else None, query, dept, cls)
        if df.empty:
            st.info("No attendance records match these filters.")
        else:
            st.dataframe(df, hide_index=True, width="stretch")
            st.caption(f"{len(df)} record(s)")
            csv = df.to_csv(index=False).encode("utf-8")
            a, b = st.columns(2)
            a.download_button("⬇ Download CSV", csv, "attendance_records.csv", "text/csv")
            if b.button("Save copy to data/exports"):
                _save_copy(df, "attendance_records.csv")

    with t2:
        d = st.date_input("Report date", date.today(), key="daily_date")
        rep = svc.reports.daily_report(d.isoformat())
        s = svc.reports.daily_summary(d.isoformat())
        if rep.empty:
            st.info("No students were enrolled on this date.")
        else:
            m = st.columns(4)
            m[0].metric("Enrolled", s["enrolled"]); m[1].metric("Present", s["present"])
            m[2].metric("Absent", s["absent"]); m[3].metric("Attendance", f"{s['percent']}%")
            st.dataframe(rep, hide_index=True, width="stretch")
            st.download_button("⬇ Download daily report", rep.to_csv(index=False).encode("utf-8"),
                               f"daily_report_{d.isoformat()}.csv", "text/csv")

    with t3:
        students = svc.students.list_students()
        if students.empty:
            st.info("No students are registered yet.")
        else:
            label = {r.student_id: f"{r.student_id} — {r.full_name}" for r in students.itertuples()}
            sid = st.selectbox("Student", list(label), format_func=label.get, key="rep_student")
            hist, stats = svc.reports.student_report(sid)
            m = st.columns(3)
            m[0].metric("Days present", stats["present"])
            m[1].metric("Session days", stats["sessions"])
            m[2].metric("Attendance", f"{stats['percent']}%")
            section("Day-by-day")
            if hist.empty:
                st.info("No session days recorded since this student registered.")
            else:
                st.dataframe(hist, hide_index=True, width="stretch")
                st.download_button("⬇ Download student report", hist.to_csv(index=False).encode("utf-8"),
                                   f"student_report_{sid}.csv", "text/csv")
