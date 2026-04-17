import streamlit as st
import numpy as np
import cv2
import time

from model import load_flood_model
from utils import preprocess_for_model, postprocess_mask, get_camera_frame

st.set_page_config(page_title="AquaFlow", layout="wide")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    .camera-card {
        background-color: rgba(255,255,255,0.03);
        padding: 12px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- LOAD MODEL ONCE ---
@st.cache_resource
def get_model():
    return load_flood_model()

model = get_model()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🛠 Camera Configuration")

    cam1_source = st.text_input("Location 1 Camera Source", value="0")
    cam2_source = st.text_input("Location 2 Camera Source", value="")
    cam3_source = st.text_input("Location 3 Camera Source", value="")
    cam4_source = st.text_input("Location 4 Camera Source", value="")

st.title("🌊 AquaFlow CCTV Flood Monitoring System")
st.write("Monitor flood-prone areas with live camera feeds and automatic flood alerts.")

# --- CAMERA SOURCES ---
cams = {
    "Location 1": cam1_source,
    "Location 2": cam2_source,
    "Location 3": cam3_source,
    "Location 4": cam4_source
}

# --- 2 x 2 GRID LAYOUT ---
top_row = st.columns(2)
bottom_row = st.columns(2)

camera_layout = {
    "Location 1": top_row[0],
    "Location 2": top_row[1],
    "Location 3": bottom_row[0],
    "Location 4": bottom_row[1]
}

# --- PLACEHOLDERS ---
placeholders = {}
metrics = {}
alerts = {}

for name, column in camera_layout.items():
    with column:
        st.markdown('<div class="camera-card">', unsafe_allow_html=True)
        st.subheader(f"📍 {name}")
        placeholders[name] = st.empty()
        metrics[name] = st.empty()
        alerts[name] = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

# --- LIVE LOOP ---
while True:
    for name, source in cams.items():

        if source.strip() == "":
            placeholders[name].info("📷 Camera not connected yet")
            metrics[name].empty()
            alerts[name].empty()
            continue

        frame = get_camera_frame(source)

        if frame is None:
            placeholders[name].warning("⚠ Unable to access camera")
            metrics[name].empty()
            alerts[name].empty()
            continue

        # --- MODEL INFERENCE ---
        input_tensor = preprocess_for_model(frame)
        prediction = model.predict(input_tensor, verbose=0)
        mask = postprocess_mask(prediction)

        # --- CREATE OVERLAY ---
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        colored_mask = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        placeholders[name].image(
            overlay_rgb,
            caption=f"{name} - Live Flood Monitoring",
            width=320
        )

        # --- FLOOD PERCENTAGE ---
        flood_pct = (np.count_nonzero(mask) / mask.size) * 100

        # --- ALERT LEVELS ---
        if flood_pct >= 50:
            metrics[name].error(f"🚨 CRITICAL FLOOD ALERT: {flood_pct:.2f}%")

            alerts[name].markdown(
                '''
                <div style="
                    background-color: rgba(255,0,0,0.2);
                    border: 2px solid red;
                    padding: 10px;
                    border-radius: 10px;
                    margin-top: 10px;
                ">
                <h4 style="color:red;">🚨 Emergency Flood Alert</h4>
                <p style="color:white;">Automatic siren triggered for this location.</p>
                </div>
                ''',
                unsafe_allow_html=True
            )

            try:
                st.audio("siren.mp3")
            except:
                pass

        elif flood_pct >= 30:
            metrics[name].warning(f"⚠ HIGH FLOOD RISK: {flood_pct:.2f}%")
            alerts[name].empty()

        elif flood_pct >= 10:
            metrics[name].info(f"⚠ Moderate Flood Risk: {flood_pct:.2f}%")
            alerts[name].empty()

        else:
            metrics[name].success(f"✅ Safe: {flood_pct:.2f}%")
            alerts[name].empty()

    time.sleep(0.2)
