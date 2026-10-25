# FaceTrack — Smart Attendance System
*AI-assisted attendance powered by OpenCV* — a college project for **Image Processing & Computer Vision**.

## 1. Overview
FaceTrack registers students with webcam face samples, recognizes them live, and records one attendance entry per student per day in SQLite. Everything runs locally on a CPU-only laptop. **It is an educational demo, not production-grade biometric authentication.**

## 2. Features
- Streamlit dashboard: Dashboard · Live Attendance · Register Student · Student Directory · Attendance Records · Image Processing Lab · About Project
- Haar Cascade face detection with adjustable scale factor, min neighbors, min size, optional histogram equalization
- LBPH face recognition with adjustable distance threshold ("Unknown" above it)
- Consecutive-frame confirmation before marking; one record per student per day (DB `UNIQUE` constraint + in-session de-duplication)
- Student directory (search, history, deactivate, delete with typed confirmation, delete face samples only)
- Filtered records, CSV export, daily and per-student reports
- Image Processing Lab comparing grayscale / equalization / blur / detection with timing graphs

## 3. Architecture
```
app.py                   Streamlit entry + sidebar navigation
config.py                paths, defaults, logging
database/db_manager.py   SQLite schema + helpers
vision/face_detector.py  Haar Cascade DETECTION (where are faces?)
vision/face_recognizer.py LBPH RECOGNITION (whose face?) + training + samples + label map
vision/camera.py         webcam open/release with clear errors
vision/image_processor.py Image Processing Lab pipeline
services/                student, attendance, report business logic (+ container.py wiring)
views/                   one module per UI page   (named `views`, not `pages`, because Streamlit
                         auto-builds a second navigation from a folder called `pages/`)
data/                    face_samples/<student_id>/*.png, models/lbph_model.yml + label_map.json, exports/
tests/                   pytest suite
```
Pipeline per frame: capture → grayscale → (equalize) → `detectMultiScale` → crop/resize/equalize face → LBPH `predict` → threshold → N consecutive frames → DB insert.

## 4. Concepts (viva material)
**Haar Cascade.** Haar features are differences between sums of pixels in adjacent light/dark rectangles (e.g. eye region darker than cheeks). An *integral image* makes any rectangle sum O(1). AdaBoost selects the most useful features and groups them into a *cascade* of stages; most windows are rejected by the first cheap stages, so the detector is fast. A window slides over an image pyramid (`scaleFactor`), and `minNeighbors` keeps only detections confirmed by overlapping neighbours.

**LBPH.** For each pixel, compare its 8 neighbours with it to form a binary code (Local Binary Pattern). The face is divided into a grid; a histogram of codes is built per cell and the histograms are concatenated. A new face is matched to the training histogram with the smallest distance. The returned value is a **distance — lower is better, not a probability**; our threshold turns it into accept/Unknown.

**Why grayscale.** Haar features and LBP compare brightness only; one channel is 3× less data and faster, and avoids colour/white-balance variation.

**Histogram equalization.** Spreads the intensity histogram using its cumulative distribution, increasing global contrast so faces in dim or washed-out images show stronger edges. It can amplify noise and doesn't fix uneven (side) lighting well (CLAHE does better).

**Gaussian blur.** Low-pass filter that suppresses noise; light blur can cut false positives, heavy blur removes the edges Haar features need, reducing detections.

## 5. Database design
`students(student_id PK, full_name, department, class_year, email, registration_date, active_status)`
`attendance(attendance_id PK, student_id FK, attendance_date, attendance_time, recognition_distance, created_at, UNIQUE(student_id, attendance_date))`
LBPH needs integer labels, so `data/models/label_map.json` maps `label → student_id`. Labels are assigned once and never reused, so the mapping does not depend on load order.
**Reporting rules:** one record per student per day; a student registered after date D is *not yet enrolled* (not absent) on D; individual percentage = days present ÷ session days (dates with any attendance) since registration.

## 6. Installation (Python 3.11)
```bash
cd facetrack
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt` pins `opencv-contrib-python<5`: OpenCV 5.x wheels no longer ship the Haar XML files.

## 7. Run
```bash
streamlit run app.py
```
Open http://localhost:8501. Tests: `python -m pytest -q`.

## 8. Usage
1. **Register Student** → fill details, tick consent, press *Validate & start face capture*. Allow camera access. Keep one face in frame; slowly turn your head slightly and vary expression until the progress bar fills (Cancel stops without saving).
2. **Live Attendance** → *Start Camera*. When the box shows your name for N consecutive frames, attendance is marked. A second appearance shows "Already marked". *Stop Camera* releases the webcam.
3. **Attendance Records** → filter, download CSV, daily/student reports.
4. **Image Processing Lab** → upload an image and tune parameters.

## 9. Troubleshooting
| Problem | Fix |
|---|---|
| "Could not open the webcam" | Close other apps using the camera; macOS: System Settings → Privacy & Security → Camera → enable your terminal/IDE, then restart it |
| `cv2.face` missing | `pip uninstall opencv-python opencv-python-headless`, then `pip install opencv-contrib-python<5` |
| Haar file not found | Reinstall OpenCV 4.x (`pip install "opencv-contrib-python<5"`) |
| Registered student shows Unknown | Raise the threshold (try 80–90), improve lighting, re-register with more varied samples |
| Wrong person recognized | Lower the threshold; add more samples; remember LBPH is weak |
| Model corrupted/missing | Student Directory → *Retrain model now* |
| Custom data folder | Set `FACETRACK_DATA_DIR` |
Logs: `data/facetrack.log`.

## 10. Limitations & future work
Haar needs near-frontal, well-lit faces; LBPH is sensitive to pose/lighting/expression and can be fooled by a printed photo (no liveness detection); small datasets, similar-looking people and a single camera limit accuracy; `putText` shows only ASCII names on the video overlay. Future: DNN detectors (YuNet/SSD) and embedding recognizers (SFace/ArcFace), CLAHE, liveness checks, multi-class sessions, authentication for admins, evaluation on a labeled test set.

## 11. Privacy
Frames are processed locally; nothing is uploaded. Face samples are stored only for registered students, after a consent notice, in `data/face_samples/<student_id>/`, and can be deleted per student. Deleting a student removes only that student's record, attendance rows, and sample folder, then retrains the model.

## 12. Viva questions
1. **Detection vs recognition?** Detection finds *that and where* a face exists; recognition decides *whose* face it is. Haar does the former, LBPH the latter.
2. **Why Haar Cascade?** Light, fast on CPU, ships with OpenCV, simple to explain. Less accurate than deep detectors.
3. **What is an integral image?** A table where each cell holds the sum of all pixels above/left, so any rectangle sum needs 4 lookups.
4. **What is a cascade?** A sequence of increasingly complex classifiers; most non-face windows are rejected early.
5. **Role of AdaBoost?** Selects the few best Haar features out of ~160k and weights weak classifiers into strong ones.
6. **`scaleFactor`?** Size step of the image pyramid; smaller = finer search, slower, more detections.
7. **`minNeighbors`?** Number of overlapping detections needed to keep a face; higher = fewer false positives, more misses.
8. **Why grayscale?** Features use intensity only; less data, faster, lighting-colour independent.
9. **What does histogram equalization do?** Flattens the intensity distribution to boost global contrast. It can amplify noise.
10. **Effect of Gaussian blur?** Removes noise but blurs edges; strong blur lowers detection.
11. **How does LBPH work?** Pixel→binary pattern vs neighbours; histograms per grid cell; nearest histogram wins.
12. **What is the LBPH confidence value?** A distance: lower = closer. Not a probability, so a threshold is tuned empirically.
13. **How is "Unknown" decided?** If best distance > threshold.
14. **Why consecutive frames?** Prevents one noisy frame from marking the wrong person.
15. **How are duplicates prevented?** `UNIQUE(student_id, attendance_date)` in SQLite plus an in-session tracker avoiding repeated writes.
16. **Why not store labels by student order?** Order changes when students are added/deleted; a persistent label map keeps stable IDs.
17. **Limitations of Haar?** Frontal faces only, lighting sensitivity, false positives, no landmarks.
18. **Limitations of LBPH?** Pose/illumination sensitivity, needs retraining when students change, spoofable by photos.
19. **How would you evaluate accuracy?** Labeled test set: precision, recall, FAR/FRR, confusion matrix, tested at several thresholds.
20. **How would you improve it?** CLAHE, DNN detection, embedding-based recognition, liveness detection.
