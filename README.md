# SpineAI: Detection and Classification of Lumbar Abnormalities Using CNN Models

An AI-powered platform that automates the analysis of lumbar spine MRI scans. It **detects** vertebrae and discs, **segments** spinal structures, and **grades** the severity of degeneration, then shows the results in a Flask web app.


## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Design](#system-design)
- [Models Used](#models-used)
- [Results](#results)
- [Web Application](#web-application)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Weights](#model-weights)
- [Setup and Run](#setup-and-run)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

Lumbar degenerative diseases such as disc degeneration and spinal deformities are a common cause of chronic back pain, and they are diagnosed from MRI scans. Manual reading by radiologists is slow, subjective and prone to human error.

SpineAI combines four CNN models (YOLOv8, Faster R-CNN, U-Net+ and DenseNet) into one pipeline that reads an MRI scan and returns annotated detections, segmentation masks, a severity grade and confidence scores.

## Problem Statement

- Manual MRI analysis is time-consuming and depends on the clinician's experience, which leads to inconsistent results and delayed diagnoses.
- Spinal disorders need specialised expertise, so accuracy is hard to keep uniform across clinical settings.
- Imaging volume keeps growing, and manual methods no longer scale.

The goal is an automated system that detects, segments and classifies lumbar abnormalities consistently and quickly.

## Key Features

- **Detection:** YOLOv8 and Faster R-CNN locate lumbar levels (L1-L2 to L5-S1) and abnormal regions
- **Segmentation:** U-Net+ with dense skip connections outlines spinal structures
- **Severity classification:** DenseNet grades degeneration as Mild, Moderate or Severe
- **Visual output:** annotated MRI with bounding boxes, masks and confidence scores
- **Web interface:** secure login, image upload and instant results
- **Logging:** user actions and predictions are recorded

## System Design

### Deployment overview
<p align="center">
  <img src="https://github.com/user-attachments/assets/29b1a305-e0a8-4efa-9e4f-9ca89a43606b" width="60%" alt="Deployment diagram">
</p>

### Project flow
<p align="center">
  <img src="https://github.com/user-attachments/assets/91a87ea7-d405-40db-b197-8cca0e5b2248" width="45%" alt="Project flowchart">
</p>

### System architecture
<p align="center">
  <img src="https://github.com/user-attachments/assets/981d5e0a-9415-47f9-836f-01b29cb0ce61" width="100%" alt="System architecture">
</p>

**Pipeline:** User input (MRI upload) -> Data preprocessing (resize to 256x256, normalisation, augmentation) -> Detection and segmentation -> Classification -> Prediction output (labels, boxes, masks) -> Visual output in the web interface.

## Models Used

| Model | Task | Notes |
|---|---|---|
| **YOLOv8** | Detection | Real-time, anchor-free detection of vertebrae and disc abnormalities |
| **Faster R-CNN** | Detection | Two-stage detector with a Region Proposal Network and multi-task loss |
| **U-Net+** | Segmentation | Encoder-decoder with dense skip connections for pixel-level masks |
| **DenseNet** | Classification | Densely connected layers for severity grading (Mild, Moderate, Severe) |

## Results

### Faster R-CNN
Predicted boxes for each lumbar level, with labels and confidence scores.

<p align="center">
  <img src="https://github.com/user-attachments/assets/d064706a-ee69-479a-b7c2-ec62f6f0458c" width="45%" alt="Faster R-CNN prediction 1">
  <img src="https://github.com/user-attachments/assets/9eec9421-1193-4672-8c23-6990c11d5a12" width="45%" alt="Faster R-CNN prediction 2">
</p>

### U-Net+
Metrics at epoch 20:

| Metric | Train | Validation |
|---|---|---|
| Loss | 0.0062 | 0.0096 |
| IoU | 0.7068 | 0.5998 |
| Dice coefficient | 0.8208 | 0.7304 |
| Pixel accuracy | 99.77% | 99.67% |

<p align="center">
  <img src="https://github.com/user-attachments/assets/8d937ab3-d8ec-41e8-be8a-b4d1bbadb698" width="70%" alt="U-Net+ training curves">
</p>

### YOLOv8
The F1-confidence curve reaches **0.94** across all classes at a confidence of 0.349.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6f83b7a2-8639-45d8-9524-d3dc923341af" width="100%" alt="YOLOv8 validation log">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c2ff601b-bb13-4ab0-9fa7-7578ddd72a73" width="48%" alt="YOLOv8 confusion matrix">
  <img src="https://github.com/user-attachments/assets/69449b2e-a196-4280-9457-79296667b384" width="48%" alt="YOLOv8 F1-confidence curve">
</p>

### DenseNet
Accuracy of **73.24%** at epoch 100, with training converging after roughly 50 epochs. "Normal/Mild" is classified best, and most errors are Severe cases predicted as Moderate or Normal/Mild.

<p align="center">
  <img src="https://github.com/user-attachments/assets/641f59b1-e7f4-4e75-840d-21154e94d90b" width="80%" alt="DenseNet training curves">
</p>

## Web Application

<p align="center">
  <img src="https://github.com/user-attachments/assets/5ae6f3e0-92de-46ed-a503-ad43dd531624" width="80%" alt="Home page">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/9749e987-2933-40ce-a402-8188ad150615" width="45%" alt="Upload page">
  <img src="https://github.com/user-attachments/assets/78bf1fe8-b9e5-4b81-8fee-63d30c5637c0" width="30%" alt="Result page">
</p>

## Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch, Ultralytics YOLOv8
- **Backend:** Flask
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Data and plotting:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Database:** MySQL
- **Tools:** VS Code, XAMPP

## Project Structure

```
Lumber-Spine-Full/
├── CODE/
│   ├── BACKEND/
│   │   └── result-yolo-unet/
│   │       └── models/        <- download weights here (see below)
│   └── FRONTEND/
├── .gitignore
└── README.md
```

## Model Weights

The trained weights (`.pth`) are too large for GitHub and are hosted on Hugging Face:

**https://huggingface.co/saisuukaanyaa/spineai-models**

Download the files into `CODE/BACKEND/result-yolo-unet/models/`.

## Setup and Run

> Adjust the file names below to match your code.

```bash
# 1. Clone the repository
git clone https://github.com/saisuukaanyaa-dev/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install torch torchvision ultralytics flask pandas numpy scikit-learn matplotlib seaborn

# 4. Download the model weights from Hugging Face (see above)

# 5. Start the app
cd CODE/BACKEND
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| Complex MRI interpretation | YOLOv8, U-Net+ and Faster R-CNN detect and segment structures automatically |
| Inconsistent diagnostic accuracy | DenseNet gives consistent, reproducible severity grades |
| Large, heavy datasets | Resizing, normalisation, augmentation and GPU training |
| Small or subtle abnormalities | Faster R-CNN for detection plus U-Net+ for fine segmentation |
| Generalisation across MRI sources | Augmentation and cross-validation |

## Future Work

The next step is publication: a research paper covering the models, methodology and results is being prepared for submission to an IEEE conference.

## References

1. Chen, K., Zheng, L., Zhao, H., & Wang, Z. (2024). Deep Learning-Based Intelligent Diagnosis of Lumbar Diseases with Multi-Angle View of Intervertebral Disc. *Mathematics*, 12(13), 2062.
2. Schneider, H., et al. (2023). Segmentation and Analysis of Lumbar Spine MRI Scans for Vertebral Body Measurements. *ESANN 2023*.
3. Zhou, Y., Liu, Y., Chen, Q., Gu, G., & Sui, X. (2019). Automatic Lumbar MRI Detection and Identification Based on Deep Learning. *PMC*.
4. Sayed, M.A., et al. (2025). Automatic detection, classification, and segmentation of sagittal MR images for diagnosing prolapsed lumbar intervertebral disc.
5. Wang, A., et al. (2025). Automated diagnosis and grading of lumbar intervertebral disc degeneration based on a modified YOLO framework. *Frontiers in Bioengineering and Biotechnology*.
6. Mousavi, A. (2024). Lumbar Spine Degenerative Classification Using YOLO v8 and DeepScoreNet.
7. Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR 2018*.
8. Lim, D.S.W., et al. (2022). Improved Productivity Using Deep Learning-assisted Reporting for Lumbar Spine MRI. *Radiology*, 305(1), 160-166.

