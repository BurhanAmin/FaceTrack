"""Live webcam attendance.

Streamlit reruns the script on every interaction, so the camera loop lives
inside one script run: Start sets a flag, the loop streams frames into
placeholders, and pressing Stop triggers a rerun that interrupts the loop. The
`finally` block guarantees the webcam is released in every case.
"""
import time
from datetime import datetime

import cv2
import streamlit as st

import config
from services.attendance_service import ALREADY, MARKED, RecognitionTracker
from vision.camera import Camera, CameraError
from vision.face_detector import DetectionParams, crop_face, draw_box
from views._ui import note, page_header, section

GREEN, AMBER, RED = (60, 160, 60), (30, 150, 220), (50, 50, 200)  # BGR


def _set(flag: bool) -> None:
    st.session_state["cam_on"] = flag


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode()  # cv2.putText cannot draw non-ASCII


def render(svc) -> None:
    page_header("Live Attendance", "Real-time face detection and recognition from the webcam")
    if svc.detector is None:
        st.error("The Haar Cascade detector could not be loaded, so live attendance is unavailable.")
        return
    if svc.reports.total_registered() == 0:
        st.warning("No students are registered. Register a student before taking attendance.")
        return
    if svc.model_error:
        st.error(svc.model_error)
    if not svc.recognizer.is_ready:
        try:
            svc.recognizer.load()
        except Exception as exc:  # noqa: BLE001 - shown to user
            st.error(str(exc))
        if not svc.recognizer.is_ready:
            st.warning("No trained recognition model found. Register a student with face samples "
                       "(or retrain from Student Directory).")
            return

    with st.expander("Detection & recognition settings", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        thr = c1.slider("Recognition threshold", 20.0, 150.0, config.DEFAULT_RECOGNITION_THRESHOLD, 1.0,
                        help="LBPH returns a DISTANCE, not a probability. Lower = more similar. A face is "
                             "accepted only if distance ≤ threshold; otherwise it is shown as Unknown.")
        frames_needed = c2.slider("Consecutive frames", 1, 30, config.DEFAULT_CONSECUTIVE_FRAMES,
                                  help="Frames in a row that must agree on the same student before marking.")
        scale = c3.slider("Scale factor", 1.05, 1.5, config.DEFAULT_SCALE_FACTOR, 0.01)
        neigh = c4.slider("Min neighbors", 1, 10, config.DEFAULT_MIN_NEIGHBORS)
        msize = c5.slider("Min face size (px)", 30, 250, config.DEFAULT_MIN_FACE_SIZE, 5)
        note("Lower the threshold to reduce wrong matches (more Unknowns); raise it if registered students "
             "are shown as Unknown. Tune it on your own camera and lighting.")

    params = DetectionParams(scale, neigh, msize, equalize=True)
    on = st.session_state.get("cam_on", False)
    b1, b2, clock_col = st.columns([1, 1, 3])
    b1.button("▶ Start Camera", on_click=_set, args=(True,), disabled=on, type="primary", width="stretch")
    b2.button("■ Stop Camera", on_click=_set, args=(False,), disabled=not on, width="stretch")
    clock = clock_col.empty()

    video = st.empty()
    mcols = st.columns(3)
    m_faces, m_known, m_marked = (c.empty() for c in mcols)
    banner = st.empty()
    section("Recent attendance activity")
    table = st.empty()

    def refresh_table():
        df = svc.attendance.recent(10)
        if df.empty:
            table.caption("No attendance recorded yet.")
        else:
            table.dataframe(df, hide_index=True, width="stretch")

    def show_metrics(f, k, m):
        m_faces.metric("Faces detected", f)
        m_known.metric("Recognized now", k)
        m_marked.metric("Marked this session", m)
        clock.markdown(f"**{datetime.now():%A, %d %B %Y · %H:%M:%S}**")

    refresh_table()
    if not on:
        video.info("Camera is off. Press **Start Camera** to begin.")
        show_metrics(0, 0, 0)
        return

    names = {r["student_id"]: r["full_name"] for r in svc.db.query("SELECT student_id, full_name FROM students")}
    tracker = RecognitionTracker(required=frames_needed)
    status: dict[str, str] = {}
    marked_count = 0
    frame_no = 0
    cam = Camera()
    try:
        cam.open()
    except CameraError as exc:
        st.session_state["cam_on"] = False
        video.error(str(exc))
        return

    try:
        while True:
            frame = cam.read()
            if frame is None:
                banner.error("Lost the camera feed. Press Stop, then Start Camera again.")
                break
            frame = cv2.flip(frame, 1)
            boxes, gray = svc.detector.detect(frame, params)
            ids = []
            for box in boxes:
                pred = svc.recognizer.predict(crop_face(gray, box), thr)
                ids.append(pred.student_id)
                if pred.student_id is None:
                    draw_box(frame, box, "Unknown", RED)
                    continue
                sid = pred.student_id
                tag = status.get(sid, "")
                draw_box(frame, box, _ascii(f"{names.get(sid, sid)} ({sid}) {tag}".strip()),
                         GREEN if tag in ("", "Marked") else AMBER)
                # Remember each student's best (lowest) distance while confirming.
                best = st.session_state.setdefault("_best", {})
                best[sid] = min(best.get(sid, 1e9), pred.distance)

            for sid in tracker.update(ids):  # at most one DB write per student per session
                res = svc.attendance.mark(sid, st.session_state["_best"].get(sid))
                if res.status == MARKED:
                    status[sid] = "Marked"
                    marked_count += 1
                    banner.success(f"✔ Attendance marked: {names.get(sid, sid)} ({sid})")
                    refresh_table()
                elif res.status == ALREADY:
                    status[sid] = "Already marked"
                    banner.info(f"Already marked today: {names.get(sid, sid)} ({sid})")

            video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            if frame_no % 3 == 0:
                show_metrics(len(boxes), sum(1 for i in ids if i), marked_count)
            frame_no += 1
            time.sleep(0.02)
    finally:
        cam.release()  # runs on Stop (script rerun), errors and normal exit
        st.session_state.pop("_best", None)
