🌊 Flood Detection and Segmentation Using Deep Learning (Attention U-Net)

This repository presents a deep learning project focused on flood area detection and segmentation using the Attention U-Net architecture. The model was trained and evaluated on the Flood Area Segmentation dataset from Kaggle to accurately identify flooded regions in satellite imagery, supporting disaster management and risk assessment.

🧠 Project Overview

Floods are among the most devastating natural disasters, affecting millions globally. Rapid and accurate flood detection is essential for disaster response and resource allocation.
This project leverages deep learning and satellite imagery to detect and segment flooded areas, using Attention U-Net for enhanced spatial focus and segmentation precision.

The model performs pixel-level classification, distinguishing flooded from non-flooded areas, aiding authorities in timely interventions and damage assessment.

📂 Dataset

Dataset Name: Flood Area Segmentation Dataset

The dataset contains images of flood hit areas and corresponding mask images showing the water region.

There are 290 images and self annoted masks. The mask images were created using Label Studio, an open source data labelling software. The task is to create a segmentation model, which can accurately segment out the water region in a given picture of a flood hit area.

Such models cane be used for flood surveys, better decision-making and planning. Because of less data, pre-trained models and data augmentation may be used.

Naviagate Dataset:
Image: Folder containing all the flood images.
Mask: Folder containing all the mask images.
metadata.csv: A csv file mapping the image name with its mask.

Link to dataset: https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation

🧩 Project Structure

<img width="794" height="421" alt="image" src="https://github.com/user-attachments/assets/39825c33-5a49-4a63-981a-aa9209e17e2e" />

⚙️ Methodology
1. Data Preprocessing

Image resizing and normalization

Mask encoding and alignment

Train-test split (80/20 ratio)

Data augmentation (flips, rotations, brightness)

2. Model Architecture

The Attention U-Net enhances the classical U-Net by adding attention gates that help the model focus on relevant spatial regions, improving flood boundary detection.

3. Training

Optimizer: Adam

Loss: Binary Cross-Entropy + Dice Loss

Metrics: IoU (Intersection over Union), Dice Coefficient, Accuracy, and F1-Score

Batch size: 8

Epochs: 50

4. Evaluation

The model was evaluated using unseen test data. Metrics were calculated to assess segmentation accuracy and visual quality of predicted masks.

📊 Results and Performance

<img width="635" height="320" alt="image" src="https://github.com/user-attachments/assets/de094843-ad45-4ea5-8351-f0276369a125" />

<img width="635" height="317" alt="image" src="https://github.com/user-attachments/assets/a5011556-e193-4571-b3cb-2c871fcedbf5" />


The Attention U-Net achieved strong segmentation performance on the test set.

Metric	Score
Accuracy	0.94
F1-Score	0.92
IoU	0.89
Dice Coefficient	0.90
📈 Training and Validation Curves

Training Accuracy vs Validation Accuracy
(Graph shows steady convergence with minimal overfitting)

Training Loss vs Validation Loss
(Loss decreases consistently across epochs)

Predicted vs Actual Masks
Visualization demonstrates precise boundary detection and clear segmentation of flooded areas.

🧾 Conclusion

This project demonstrates the capability of Attention U-Net in accurately segmenting flooded regions from satellite images.
The model’s strong IoU and Dice scores indicate its potential for use in real-world flood monitoring systems.

🧭 Recommendations for Improvement

Integrate multispectral data (e.g., Sentinel-1 SAR) for improved water-body distinction.

Employ transfer learning from geospatial pre-trained models.

Implement real-time flood monitoring via web deployment or API integration.

Experiment with transformer-based architectures (e.g., SegFormer, Swin-UNet) for larger-scale applications.

🎯 Essence of the Project

This project underscores the power of deep learning in environmental applications — showcasing how AI can support climate resilience and disaster response.
By automating flood detection, the project aims to assist governments, NGOs, and urban planners in mitigating the impact of floods through data-driven decision-making.

🧰 Tools and Technologies

Programming Language: Python

Libraries: TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Scikit-learn

Architecture: Attention U-Net

Environment: Jupyter Notebook / Google Colab

Visualization: Matplotlib & Seaborn

🌍 Future Scope

Integration with GIS platforms for flood impact mapping

Use of real-time satellite feeds for live flood detection

Deployment as a web-based interactive tool for policymakers

🙏 Acknowledgments

GeoDev YouTube Channel — for detailed tutorial videos that guided this project.

Kaggle Dataset Contributors — for the Flood Area Segmentation dataset.

TensorFlow/Keras Community — for open-source deep learning resources.

