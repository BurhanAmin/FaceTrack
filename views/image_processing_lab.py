import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from vision.image_processor import ImageLoadError, load_image_bytes, run_lab
from views._ui import note, page_header, section


def render(svc) -> None:
    page_header("Image Processing Lab", "Compare preprocessing stages and Haar Cascade detection side by side")
    if svc.detector is None:
        st.error("The Haar Cascade detector could not be loaded.")
        return
    up = st.file_uploader("Upload an image (JPG / PNG)", type=["jpg", "jpeg", "png"])
    c = st.columns(5)
    scale = c[0].slider("Scale factor", 1.05, 1.5, 1.2, 0.01)
    neigh = c[1].slider("Min neighbors", 1, 10, 5)
    msize = c[2].slider("Min face size (px)", 20, 300, 40, 5)
    blur_k = c[3].slider("Gaussian kernel", 1, 31, 5, 2)
    eq_on = c[4].toggle("Histogram equalization", value=True)

    if up is None:
        st.info("Upload a photo containing faces (a group photo works well) to see each stage.")
    else:
        try:
            img = load_image_bytes(up.getvalue())
        except ImageLoadError as exc:
            st.error(str(exc))
            return
        r = run_lab(svc.detector, img, scale, neigh, msize, blur_k, eq_on)
        eq_label = "6. Detection after equalization" if eq_on else "6. Detection (equalization OFF)"
        stages = [("1. Original", r.original_rgb), ("2. Grayscale", r.gray), ("3. Histogram equalized", r.equalized),
                  ("4. Gaussian blur", r.blurred), ("5. Detection on grayscale", r.plain.annotated_rgb),
                  (eq_label, r.equalized_det.annotated_rgb)]
        for row in range(2):
            cols = st.columns(3)
            for col, (title, im) in zip(cols, stages[row * 3: row * 3 + 3]):
                col.image(im, caption=title, width="stretch")

        m = st.columns(4)
        m[0].metric("Faces (grayscale)", r.plain.count)
        m[1].metric("Time (grayscale)", f"{r.plain.ms:.1f} ms")
        m[2].metric("Faces (equalized)" if eq_on else "Faces (eq. OFF)", r.equalized_det.count)
        m[3].metric("Time (equalized)" if eq_on else "Time (eq. OFF)", f"{r.equalized_det.ms:.1f} ms")

        section("Comparison graphs")
        g1, g2, g3 = st.columns(3)
        names = ["Grayscale", "Equalized" if eq_on else "Eq. OFF"]
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.bar(names, [r.plain.count, r.equalized_det.count], color=["#8AA4C8", "#1F4E8C"])
        ax.set_title("Faces detected"); ax.yaxis.get_major_locator().set_params(integer=True)
        g1.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.bar(names, [r.plain.ms, r.equalized_det.ms], color=["#9CCDB5", "#1E8E5A"])
        ax.set_title("Processing time (ms)")
        g2.pyplot(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.plot(r.gray_hist, label="grayscale", color="#8AA4C8")
        ax.plot(r.eq_hist, label="equalized", color="#1F4E8C")
        ax.set_title("Intensity histograms"); ax.legend(fontsize=7)
        g3.pyplot(fig); plt.close(fig)
        note("A higher face count is NOT automatically better: extra boxes may be false positives. "
             "Judging accuracy properly needs labeled test images (ground-truth boxes) to compute "
             "precision and recall. Timings include only detection, on this machine.")

    section("What each operation does (viva notes)")
    with st.expander("Grayscale conversion"):
        st.write("Replaces 3 colour channels with one intensity value per pixel (≈0.299R + 0.587G + 0.114B). "
                 "Haar features compare brightness of rectangular regions, so colour is unnecessary; it cuts "
                 "data to a third and makes detection faster.")
    with st.expander("Histogram equalization"):
        st.write("Redistributes pixel intensities so the cumulative histogram becomes roughly linear, "
                 "stretching contrast. It helps under dim or washed-out lighting, but it can also amplify "
                 "noise and may not help (or may hurt) already well-lit images.")
    with st.expander("Gaussian blur"):
        st.write("Convolves the image with a Gaussian kernel to suppress high-frequency noise. Mild blur can "
                 "reduce false detections caused by noise; heavy blur erases the eye/nose edges that Haar "
                 "features rely on. It is shown here for comparison and is not applied before detection in "
                 "the live pipeline.")
    with st.expander("Haar Cascade detection and its parameters"):
        st.write("A window slides over the image at several scales. Each stage of the cascade evaluates Haar "
                 "features (light/dark rectangle differences) computed fast from an integral image and rejects "
                 "non-faces early. **Scale factor**: how much the window grows per pass (smaller = finer, "
                 "slower). **Min neighbors**: how many overlapping detections must agree (higher = fewer false "
                 "positives, may miss faces). **Min face size**: ignore windows smaller than this.")
