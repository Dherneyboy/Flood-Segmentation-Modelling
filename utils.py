import ee
import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
import pandas as pd

def fetch_sentinel_rgb(lat, lon):
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(500).bounds()
    
    image = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(point) \
        .filterDate('2026-01-01', '2026-04-14') \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() \
        .select(['B4', 'B3', 'B2'])
    
    # Use sampleRectangle: This bypasses the Map Thumbnailer and gives you
    # the raw spectral values in a structured dictionary.
    data = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    
    # Extract the arrays directly from the dictionary
    b4 = np.array(data['properties']['B4'])
    b3 = np.array(data['properties']['B3'])
    b2 = np.array(data['properties']['B2'])
    
    # Stack them into an RGB image
    img = np.dstack((b4, b3, b2))
    
    # Normalize: Sentinel SR values are typically 0-10000.
    # We map this to 0-255 for OpenCV/Model compatibility.
    img = np.clip(img / 3000 * 255, 0, 255).astype(np.uint8)
    
    # Resize to the input size your model expects (e.g., 512x512)
    return cv2.resize(img, (512, 512))

def preprocess_for_model(image):
    """Resizes and normalizes the image to match model input."""
    img = cv2.resize(image, (128, 128))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0) # Add batch dimension

def postprocess_mask(prediction):
    """Converts model output probability map into a binary mask."""
    mask = (prediction[0] > 0.50).astype(np.uint8) * 255
    return mask