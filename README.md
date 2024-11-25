# student-behavior-detection
A project to detect and classify student behaviors in classrooms using YOLOv8.

This repository contains a computer vision project aimed at detecting and classifying student behaviors in classroom settings using YOLO-based object detection models. The project leverages the students-behaviors-detection dataset, available on Roboflow, and integrates state-of-the-art techniques for behavior recognition and analysis.

Project Overview
Purpose
The goal of this project is to automate the detection of classroom behaviors to enhance educational monitoring and improve teaching strategies. The system detects various behaviors such as reading, writing, raising hands, following instructions, and more, using a YOLO-based object detection model.

Features
11 Behavior Classes:
Closed-Book
Electronic-Book
No-Book
Opened-Book
Raising-Hand
Student-Answers
Student-Reads
Student-Writes
Teacher-Explains
Teacher-Follows-up-Students
Worksheet
Pre-trained Model: Utilizes YOLOv8n or other lightweight YOLO models for quick inference.
Visualization Tools: Confusion matrices, training metrics, and class distributions.
Customizable Framework: Easy to adapt to other classroom datasets or behaviors.

Dataset
Source
The dataset used for training and validation is hosted on Roboflow.

Dataset Link: students-behaviors-detection dataset

Key Details
Format: YOLOv5/YOLOv8 format.
Annotations: Bounding boxes for each behavior class.
Structure: Contains train, valid, and test splits.
Images: Annotated classroom scenes.
Folder Structure
The project is organized as follows:

plaintext
Copy code
student-behavior-detection/
├── src/                # Python scripts for training, evaluation, and inference
│   ├── student-behavior-detection.py  # Main script
├── data/               # Placeholder for small sample datasets or links
│   └── README.md       # Instructions for downloading datasets
├── results/            # Outputs like confusion matrices, training metrics, etc.
│   ├── training_metrics.png
│   ├── confusion_matrix.png
├── models/             # Pre-trained YOLO weights or lightweight placeholders
│   ├── yolov8n.pt
├── docs/               # Documentation or additional notes
│   ├── evaluation_metrics.csv
│   ├── usage_guide.md
├── README.md           # Project documentation

Setup and Installation
Prerequisites
Python 3.8+
PyTorch
NVIDIA CUDA (for GPU support)
Git
Installation Steps
Clone this repository:

bash
Copy code
git clone https://github.com/Enock02333/student-behavior-detection.git
cd student-behavior-detection
Create a virtual environment and activate it:

bash
Copy code
python -m venv yolov8-env
source yolov8-env/bin/activate   # On Windows: yolov8-env\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Download the dataset from Roboflow:

Navigate to the Roboflow dataset page.
Export it in the YOLOv8 format.
Place the dataset in the data/ folder.
Prepare the dataset path in data.yaml: Ensure that train, val, and test paths in the dataset configuration file point to the correct folders.

Training
To train the model:

bash
Copy code
python src/student-behavior-detection.py
Key Parameters:
epochs=100: Number of training epochs.
batch_size=8: Batch size.
imgsz=416: Image size.
weights='models/yolov8n.pt': Pre-trained YOLO weights.
Evaluation
To evaluate the trained model:

bash
Copy code
python src/student-behavior-detection.py --evaluate
Outputs:

Training Metrics: Saved in results/training_metrics.png.
Confusion Matrix: Saved in results/confusion_matrix.png.
Evaluation Metrics: Saved as docs/evaluation_metrics.csv.
Testing and Inference
To test the model or perform inference on new images:

Place images in a folder (e.g., test_images/).
Run the inference script:
bash
Copy code
python src/student-behavior-detection.py --inference --images test_images/
Outputs will be saved in results/.
Usage Guide
Real-Time Monitoring
This framework can be extended for real-time video monitoring by integrating with live camera feeds.

Integration with Larger Systems
You can integrate this detection framework with educational management systems for automated reporting and analytics.

Future Work
Integrating temporal analysis to account for behavior changes over time.
Expanding the dataset to include diverse classroom settings globally.
Enhancing model accuracy by incorporating ensemble methods.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Dataset: Roboflow - students-behaviors-detection dataset.
YOLO Framework: Ultralytics.

