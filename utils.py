import numpy as np
import cv2
import streamlit as st
import cv2

@st.cache_resource
def open_camera(source):
   
    if str(source).isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # FIX for Windows

    return cap

def get_camera_frame(source):

    try:
        cap = open_camera(source)

        if not cap.isOpened():
            return None

        ret, frame = cap.read()

        if not ret or frame is None:
            return None

        frame = cv2.resize(frame, (512, 512))
        return frame

    except Exception:
        return None


def preprocess_for_model(image):
    """Resize and normalize image for model input."""
    img = cv2.resize(image, (128, 128))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


def postprocess_mask(prediction):
    """
    Converts model output probability map into a cleaner binary mask.
    """

    mask = (prediction[0] > 0.85).astype(np.uint8) * 255

    mask = mask.squeeze()

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area > 500:
            cleaned_mask[labels == i] = 255

    return cleaned_mask
