import streamlit as st
import numpy as np
import cv2
import os
from pathlib import Path
from src.detection import load_yolo, run_yolo, extract_detections
from src.segmentation import load_unet, run_unet
from src.heatmap import compute_occlusion_heatmap
from src.reporting import generate_ct_report
from src.utils import read_image_bytes_as_bgr, draw_boxes_on_image, ensure_dirs, save_image, to_bgr_rgb

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"
OUTPUT_DIR = BASE / "outputs"
ensure_dirs([OUTPUT_DIR / "annotated", OUTPUT_DIR / "heatmaps", OUTPUT_DIR / "reports"])

st.set_page_config(page_title="KidneyStone-X", layout="centered")

st.title("KidneyStone-X")
st.markdown("Clinical Decision Support for Renal Calculi Detection")

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    yolo = load_yolo(MODELS_DIR / "YOLO_stone_final_v3.pt")
    unet = load_unet(MODELS_DIR / "UNet_kidney_v5.pth")
    return yolo, unet

yolo_model, unet_model = load_models()

# -------------------------
# UI: Upload
# -------------------------
uploaded = st.file_uploader("Upload axial CT slice (png/jpg/jpeg)", type=["png","jpg","jpeg"])
show_unet_overlay = st.checkbox("Show kidney segmentation overlay (UNet)", value=True)

if uploaded:
    orig_bgr = read_image_bytes_as_bgr(uploaded.read())
    H, W = orig_bgr.shape[:2]
    st.image(to_bgr_rgb(orig_bgr), caption="Uploaded image", use_column_width=True)

    if st.button("Analyze Scan"):
        with st.spinner("Running detection and analysis (CPU optimized)..."):
            # YOLO
            result = run_yolo(yolo_model, orig_bgr)  
            boxes, scores = extract_detections(result)

            # UNet segmentation (optional)
            mask_resized = None
            if unet_model is not None:
                try:
                    mask_resized = run_unet(unet_model, orig_bgr)
                except Exception as e:
                    mask_resized = None
                    st.warning("UNet failed: " + str(e))

            # compute measurement metrics
            stone_count = len(boxes)
            areas = []
            intensities = []
            gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
            for b in boxes:
                x1,y1,x2,y2 = map(int, b)
                area = max(0, x2-x1) * max(0, y2-y1)
                areas.append(area)
                crop = gray[y1:y2, x1:x2]
                mean_i = float(np.mean(crop)) if crop.size>0 else 0.0
                intensities.append(mean_i)
            avg_area = float(np.mean(areas)) if areas else 0.0
            mean_intensity = float(np.mean(intensities)) if intensities else 0.0
            side = "Unknown"
            if stone_count > 0:
                cx = ((boxes[0][0] + boxes[0][2]) / 2.0)
                side = "Left" if cx < (W/2) else "Right"

            # Heatmap (occlusion)
            if stone_count == 0:
                st.error("No detections found — cannot run heatmap.")
                heatmap_overlay = None
            else:
                heatmap_overlay = compute_occlusion_heatmap(orig_bgr, boxes[0], yolo_model)

            # Create overlays
            annotated = draw_boxes_on_image(orig_bgr.copy(), boxes, scores)
            if mask_resized is not None and show_unet_overlay:
                color = (0,0,255)
                overlay_color = orig_bgr.copy()
                overlay_color[mask_resized==255] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay_color, 0.3, 0)

            # Save outputs
            stem = "uploaded_scan"
            annotated_path = OUTPUT_DIR / "annotated" / f"{stem}_annotated.png"
            heatmap_path = OUTPUT_DIR / "heatmaps" / f"{stem}_heatmap.png"
            report_path = OUTPUT_DIR / "reports" / f"{stem}_report.txt"
            save_image(annotated_path, annotated)
            if heatmap_overlay is not None:
                save_image(heatmap_path, heatmap_overlay)
            report_text = generate_ct_report(stone_count, avg_area, mean_intensity, side)
            with open(report_path, "w") as f:
                f.write(report_text)

        # Tabs for results
        tab1, tab2, tab3 = st.tabs(["Detection", "Heatmap", "Report"])

        with tab1:
            st.header("Detection")
            st.image(to_bgr_rgb(annotated), use_column_width=True)
            st.write(f"Detected stones: {stone_count}")
            if stone_count > 0:
                st.write("Bounding boxes (x1,y1,x2,y2) and confidence:")
                for i, (b, s) in enumerate(zip(boxes, scores)):
                    st.write(f"stone {i+1}: {list(map(int,b))} conf={s:.3f}")

            st.download_button("Download Annotated Image", annotated_path.read_bytes(), file_name=annotated_path.name)

        with tab2:
            st.header("Occlusion Sensitivity Heatmap")
            if heatmap_overlay is not None:
                st.image(to_bgr_rgb(heatmap_overlay), use_column_width=True)
                st.download_button("Download Heatmap", heatmap_path.read_bytes(), file_name=heatmap_path.name)
            else:
                st.info("Heatmap not available (no detection)")

        with tab3:
            st.header("Diagnostic Report")
            st.code(report_text)
            st.download_button("Download Report (.txt)", report_text, file_name=report_path.name, mime="text/plain")
