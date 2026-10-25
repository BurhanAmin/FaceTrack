import streamlit as st

import config
from views._ui import note, page_header, section


def render(svc) -> None:
    page_header("About Project", config.APP_SUBTITLE)
    st.markdown("""
**FaceTrack** is a college project for *Image Processing and Computer Vision* that marks attendance using a webcam.

### Pipeline
1. **Capture** a frame from the webcam (OpenCV `VideoCapture`).
2. **Pre-process**: convert to grayscale and (optionally) equalize the histogram.
3. **Detect** faces with a **Haar Cascade** (`detectMultiScale`) — *where* are the faces?
4. **Recognize** each cropped face with **LBPH** — *whose* face is it?
5. **Decide**: accept only if the LBPH distance ≤ threshold for N consecutive frames.
6. **Record** one attendance row per student per day in SQLite.

### Detection vs recognition
Detection finds faces of anyone; recognition compares a detected face against registered students.
They are separate modules (`vision/face_detector.py`, `vision/face_recognizer.py`).
""")
    section("Privacy & limitations")
    note("All processing is local; no image leaves this computer. Face samples exist only for registered "
         "students and can be deleted from the Student Directory. This is an educational demo, NOT "
         "production-grade biometric authentication: Haar + LBPH can be fooled by photos, are sensitive to "
         "lighting and pose, and can misidentify people.")
    section("System status")
    c1, c2, c3 = st.columns(3)
    c1.metric("Haar Cascade", "Loaded" if svc.detector else "Unavailable")
    c2.metric("LBPH model", "Trained" if svc.recognizer.is_ready else "Not trained")
    c3.metric("OpenCV face module", "Available" if hasattr(__import__("cv2"), "face") else "Missing")
