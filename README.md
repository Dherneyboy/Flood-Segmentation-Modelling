🌊 AquaFlow: 

AI-Powered Flood Segmentation & Monitoring
This repository presents AquaFlow, a professional-grade hydroinformatics tool designed for precise flood area detection and segmentation. By leveraging an Attention U-Net architecture, the project provides a sophisticated bridge between raw satellite spectral data and actionable flood risk intelligence.

🧠 Project Overview

Floods are among the most devastating natural disasters globally. Rapid and accurate detection is essential for disaster response and resource allocation. AquaFlow solves this through:

Tiered Data Architecture: Supports both Live Sentinel-2 (Copernicus) near-real-time satellite feeds and Local High-Resolution Uploads (e.g., Planet/Maxar imagery).

High-Fidelity Processing: Utilizes direct spectral band extraction (sampleRectangle) to bypass cartographic map artifacts, ensuring the model analyzes pure, unadulterated ground reflectance values.

Attention-Based Segmentation: Employs the Attention U-Net architecture to focus on relevant spatial features, significantly improving boundary detection in complex urban and rural landscapes.

🛰 The AquaFlow Technical Pipeline

Unlike traditional thumbnail-based applications, AquaFlow utilizes a data-first approach:

Data Acquisition: Automated extraction of B4, B3, and B2 bands directly from the Copernicus Sentinel-2 database.

Spectral Normalization: Custom contrast stretching and normalization layers to prepare raw satellite data for neural network inference.

Inference Engine: A trained Attention U-Net model performs pixel-wise classification, distinguishing inundation zones from permanent water bodies and urban surfaces.

Dashboard Visualization: An interactive Streamlit interface providing side-by-side comparison of input imagery and AI-generated flood masks.

🧩 Methodology & Training

Architecture: Attention U-Net (with spatial attention gates).

Training: Adam Optimizer, Binary Cross-Entropy + Dice Loss.

Data Strategy: Data augmentation (flips, rotations, brightness) to handle limited real-world datasets.

Metrics: Achieved significant performance: 0.94 Accuracy, 0.92 F1-Score, 0.89 IoU.


🧭 Roadmap & Future Scope

Multispectral Fusion: Integrating Sentinel-1 (SAR) data to enable flood detection through cloud cover.

Topographic Constraints: Implementing elevation masking to prevent urban infrastructure (like rooftops) from being misidentified as flooded zones.

GIS Integration: Direct export capabilities to standard GIS platforms for professional impact mapping.

🧰 Tools and Technologies

Languages: Python

Deep Learning: TensorFlow, Keras

Geospatial Processing: Google Earth Engine (GEE), OpenCV, NumPy, Pandas

Web Framework: Streamlit

Visualization: Matplotlib, Seaborn

🙏 Acknowledgments


Kaggle Contributors: For the Flood Area Segmentation dataset.

GeoDev: For foundational tutorials on deep learning applications in geospatial analysis.

Open Source Community: For the invaluable TensorFlow/Keras and Streamlit ecosystems.
