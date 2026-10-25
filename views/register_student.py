"""Student registration with webcam face-sample capture."""
import time

import cv2
import streamlit as st

import config
from services.student_service import StudentError, StudentService
from vision.camera import Camera, CameraError
from vision.face_detector import DetectionParams, crop_face, draw_box
from vision.face_recognizer import RecognizerError
from views._ui import note, page_header


def _cancel() -> None:
    st.session_state["reg_capturing"] = False
    st.session_state["reg_cancelled"] = True


def render(svc) -> None:
    page_header("Register Student", "Enter details, consent, then capture face samples from the webcam")
    if svc.detector is None:
        st.error("The Haar Cascade detector could not be loaded, so registration is unavailable.")
        return

    note("Privacy notice: registering a student stores their details and several cropped grayscale face "
         "images on THIS computer (data/face_samples/). Nothing is uploaded anywhere. Samples are used "
         "only to train the attendance recognizer and can be deleted from the Student Directory.")

    with st.form("register_form"):
        c1, c2 = st.columns(2)
        sid = c1.text_input("Student ID *", key="reg_id")
        name = c2.text_input("Full name *", key="reg_name")
        dept = c1.text_input("Course / department *", key="reg_dept")
        cls = c2.text_input("Year / class *", key="reg_cls")
        email = st.text_input("Email (optional)", key="reg_email")
        n_samples = st.slider("Face samples to capture", 10, 40, config.SAMPLES_PER_STUDENT,
                              help="Slowly turn your head slightly and vary your expression during capture.")
        consent = st.checkbox("The student has been informed and consents to their face images being "
                              "stored locally for attendance recognition.", key="reg_consent")
        submitted = st.form_submit_button("Validate & start face capture", type="primary")

    if st.session_state.pop("reg_cancelled", False):
        st.info("Capture cancelled. Nothing was saved — adjust the form and try again.")
    if not submitted:
        return
    errors = StudentService.validate(sid, name, dept, cls, email)
    if errors:
        for e in errors:
            st.error(e)
        return
    if svc.students.exists(sid):
        st.error(f"Student ID '{sid.strip()}' is already registered. Use a different ID.")
        return
    if not consent:
        st.error("Consent is required before face images can be captured.")
        return

    sid = sid.strip()
    st.session_state["reg_capturing"] = True
    st.button("✖ Cancel capture", on_click=_cancel)
    st.info("Look at the camera. Samples are accepted only when exactly ONE face is detected. "
            "Move your head slightly and change expression between samples.")
    video, progress, msg = st.empty(), st.progress(0.0), st.empty()
    params = DetectionParams(config.DEFAULT_SCALE_FACTOR, config.DEFAULT_MIN_NEIGHBORS,
                             config.DEFAULT_MIN_FACE_SIZE, True)
    samples = []
    cam = Camera()
    try:
        cam.open()
    except CameraError as exc:
        video.error(str(exc))
        return

    deadline, last = time.time() + 90, 0.0
    try:
        while len(samples) < n_samples and time.time() < deadline:
            frame = cam.read()
            if frame is None:
                msg.error("Lost the camera feed. Nothing was saved; please try again.")
                return
            frame = cv2.flip(frame, 1)
            boxes, gray = svc.detector.detect(frame, params)
            if len(boxes) == 0:
                msg.warning("No face detected — face the camera in good light.")
            elif len(boxes) > 1:
                msg.warning(f"{len(boxes)} faces detected — only one person may be in frame.")
                for b in boxes:
                    draw_box(frame, b, "rejected", (50, 50, 200))
            else:
                draw_box(frame, boxes[0], f"{len(samples)}/{n_samples}")
                if time.time() - last >= 0.4:  # spacing gives natural pose variation
                    samples.append(crop_face(gray, boxes[0]))
                    last = time.time()
                    msg.success(f"Captured sample {len(samples)} of {n_samples}")
                    progress.progress(len(samples) / n_samples)
            video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            time.sleep(0.02)
    finally:
        cam.release()

    # Reached only if capture ended on its own (cancel interrupts the script before this point).
    st.session_state["reg_capturing"] = False
    if len(samples) < config.MIN_SAMPLES_TO_TRAIN:
        st.error(f"Only {len(samples)} usable sample(s) were captured (need at least "
                 f"{config.MIN_SAMPLES_TO_TRAIN}). Nothing was saved — check lighting and try again.")
        return
    if len(samples) < n_samples:
        st.warning(f"Capture timed out with {len(samples)} of {n_samples} samples; registering with those.")
    try:
        svc.students.add_student(sid, name, dept, cls, email)
        try:
            for img in samples:
                svc.recognizer.save_sample(sid, img)
            result = svc.students.retrain()
        except (RecognizerError, OSError):
            svc.recognizer.delete_samples(sid)
            svc.students.remove_student_row_only(sid)
            raise
    except StudentError as exc:
        st.error(str(exc))
        return
    except (RecognizerError, OSError) as exc:
        st.error(f"Registration rolled back: {exc}")
        return
    video.empty()
    st.success(f"✔ {name.strip()} ({sid}) registered with {len(samples)} face samples. "
               f"Model retrained on {result['images']} images from {result['students']} student(s).")
