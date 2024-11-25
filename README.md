# Student Behavior Detection

A computer vision project aimed at detecting and classifying student behaviors in classroom settings using YOLO-based object detection models. This repository demonstrates how AI can enhance educational monitoring and teaching strategies.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Future Work](#future-work)

---

## Overview

### Purpose

This project automates the detection of various student behaviors in classrooms using object detection models. It can recognize behaviors like reading, writing, raising hands, and answering questions, enabling educators to:

- Monitor classroom activities.
- Enhance teaching strategies based on behavioral analysis.
- Identify engagement levels of students.

### Key Features

- **11 Behavior Classes:**
  - Closed-Book
  - Electronic-Book
  - No-Book
  - Opened-Book
  - Raising-Hand
  - Student-Answers
  - Student-Reads
  - Student-Writes
  - Teacher-Explains
  - Teacher-Follows-up-Students
  - Worksheet
- **YOLO-based Models:** Uses lightweight pre-trained YOLO models like YOLOv8n for fast and efficient detection.
- **Visualization Outputs:** Generates metrics like confusion matrices and precision-recall charts.

---

## Dataset

### Source

The dataset is publicly available on Roboflow:  
[**students-behaviors-detection dataset**](https://roboflow.com/dataset/students-behaviors-detection)

### Details

- **Format:** YOLOv8-compatible.
- **Annotations:** Includes bounding boxes for 11 behavior classes.
- **Structure:** Split into train, validation, and test sets.

---

## Project Structure

```plaintext
student-behavior-detection/
├── src/                # Python scripts for training, evaluation, and inference
│   ├── student-behavior-detection.py  # Main detection script
├── data/               # Placeholder or README with dataset download instructions
├── results/            # Outputs like training metrics and confusion matrices
│   ├── confusion_matrix.png
│   ├── training_metrics.png
├── models/             # YOLO pre-trained weights or placeholder links
│   ├── yolov8n.pt
├── docs/               # Documentation or extended notes
│   ├── evaluation_metrics.csv
├── README.md           # Project documentation
```

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8 or higher
- NVIDIA CUDA (for GPU acceleration)
- Git

---

### Installation Steps

1. **Clone this repository:**
   ```bash
   git clone https://github.com/Enock02333/student-behavior-detection.git
   cd student-behavior-detection
   ```

2. **Create a virtual environment:**
    ```bash
    python -m venv yolov8-env
    source yolov8-env/bin/activate  # Windows: yolov8-env\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**
    - Visit the Roboflow dataset page.
    - Export the dataset in YOLOv8 format.
    - Place the dataset in the data/ folder.

---

## Training
To train the YOLOv8 model, use the following command:
    ```bash
    python src/student-behavior-detection.py --train
    ```
### Key Parameters:
  - epochs=100: Number of epochs.
  - batch_size=8: Training batch size.
  - imgsz=416: Image size.
  - weights=models/yolov8n.pt: Path to pre-trained weights.

---

## Evaluation
Evaluate the model's performance:
   ```bash
   python src/student-behavior-detection.py --evaluate
   ```

### Outputs:
  - **Confusion Matrix:** results/confusion_matrix.png
  - **Precision-Recall Curves:** Saved in the results/ folder.

---

## Inference
Perform inference on new images:
   ```bash
   python src/student-behavior-detection.py --inference --images test_images/
   ```
    
The processed images will be saved in the results/ directory with bounding boxes and class labels.

---

## Future Work
  - Expand the dataset to include diverse classroom environments.
  - Integrate real-time video monitoring for live classroom tracking.
