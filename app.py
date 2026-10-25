"""FaceTrack — Smart Attendance System (Streamlit entry point).

Run with:  streamlit run app.py
"""
import streamlit as st

import config
from database import DatabaseError
from services.container import build_services
from views import (about, attendance_records, dashboard, image_processing_lab, live_attendance,
                   register_student, student_directory)
from views._ui import load_css

st.set_page_config(page_title="FaceTrack", page_icon="🎓", layout="wide")

ICONS = {
    "Dashboard": "📊", "Live Attendance": "📹", "Register Student": "➕", "Student Directory": "👥",
    "Attendance Records": "🗂️", "Image Processing Lab": "🔬", "About Project": "ℹ️",
}

PAGES = {
    "Dashboard": dashboard,
    "Live Attendance": live_attendance,
    "Register Student": register_student,
    "Student Directory": student_directory,
    "Attendance Records": attendance_records,
    "Image Processing Lab": image_processing_lab,
    "About Project": about,
}


@st.cache_resource(show_spinner="Starting FaceTrack…")
def get_services():
    return build_services()


def main() -> None:
    load_css()
    try:
        svc = get_services()
    except DatabaseError as exc:
        st.error(str(exc))
        st.stop()

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Dashboard"
    with st.sidebar:
        st.markdown(
            '<div class="ft-logo"><div class="ft-logo-mark">F</div><div>'
            '<div class="ft-brand">FaceTrack</div><div class="ft-brand-sub">Smart Attendance System</div>'
            '</div></div><div class="ft-nav-label">MENU</div>', unsafe_allow_html=True)
        st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed",
                 format_func=lambda p: f"{ICONS[p]}  {p}")
        total = svc.reports.total_registered()
        det_ok, model_ok = svc.detector is not None, svc.recognizer.is_ready
        st.markdown(
            '<div class="ft-nav-label" style="margin-top:1.4rem">SYSTEM STATUS</div>'
            f'<div class="ft-status"><span class="dot {"on" if det_ok else "off"}"></span>Haar detector</div>'
            f'<div class="ft-status"><span class="dot {"on" if model_ok else "off"}"></span>LBPH model</div>'
            f'<div class="ft-status"><span class="dot on"></span>{total} student(s) registered</div>'
            f'<div class="ft-side-foot">{config.APP_SUBTITLE}<br>Local processing · Educational project</div>',
            unsafe_allow_html=True)

    if svc.detector_error:
        st.error(f"Face detector unavailable: {svc.detector_error}")
    PAGES[st.session_state["nav"]].render(svc)


main()
