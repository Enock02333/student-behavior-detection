import os
import yaml
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from pathlib import Path

class StudentBehaviorDetector:
    def __init__(self, data_yaml_path):
        """Initialize the detector with dataset configuration."""
        self.data_yaml_path = data_yaml_path
        with open(data_yaml_path, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.class_names = self.dataset_config['names']
        self.num_classes = self.dataset_config['nc']
        self.model = None
        self.results = None
        
    def train_model(self, epochs=100, imgsz=416, batch_size=8):
        """Train the YOLOv8 model."""
        # Initialize YOLOv8 model
        self.model = YOLO('yolov8l.pt')  # Load pretrained YOLOv11 model

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        
        # Train the model
        self.results = self.model.train(
            data=self.data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            amp=True,
            workers=2,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=True
        )
        
    def evaluate_model(self):
        """Evaluate the trained model on validation set."""
        if self.model is None:
            raise ValueError("Model needs to be trained first!")
            
        # Run validation
        val_results = self.model.val()
        return val_results
    
    def plot_training_metrics(self):
        """Plot training metrics."""
        if self.results is None:
            raise ValueError("No training results available!")
        
        metrics = ['box_loss', 'cls_loss', 'dfl_loss', 'precision', 'recall', 'mAP50', 'mAP50-95']
        fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        for idx, metric in enumerate(metrics):
            axs[idx].plot(self.results.results_dict[metric], label=f'Training {metric}')
            axs[idx].set_title(f'{metric} over epochs')
            axs[idx].set_xlabel('Epoch')
            axs[idx].set_ylabel(metric)
            axs[idx].legend()
            
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
    def plot_confusion_matrix(self, val_pred_path):
        """Plot confusion matrix from validation predictions."""
        true_labels = []
        pred_labels = []
        
        # Load validation predictions
        val_pred = Path(val_pred_path)
        for txt_file in val_pred.glob('*.txt'):
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    true_class = int(line.split()[0])
                    pred_class = int(float(line.split()[1]))
                    true_labels.append(true_class)
                    pred_labels.append(pred_class)
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def plot_class_distribution(self):
        """Plot distribution of classes in training set."""
        train_path = Path(self.dataset_config['train'])
        class_counts = {name: 0 for name in self.class_names}
        
        # Count instances of each class
        for label_file in (train_path / 'labels').glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    class_idx = int(line.split()[0])
                    class_counts[self.class_names[class_idx]] += 1
        
        # Create bar plot
        plt.figure(figsize=(15, 6))
        sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
        plt.title('Class Distribution in Training Set')
        plt.xticks(rotation=45)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        
    def export_metrics_report(self, val_results):
        """Export detailed metrics report to CSV."""
        metrics_dict = {
            'Class': self.class_names,
            'Precision': val_results.results_dict['precision'],
            'Recall': val_results.results_dict['recall'],
            'mAP50': val_results.results_dict['map50'],
            'mAP50-95': val_results.results_dict['map50-95'],
            'F1-Score': [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                        for p, r in zip(val_results.results_dict['precision'], 
                                      val_results.results_dict['recall'])]
        }
        
        df = pd.DataFrame(metrics_dict)
        df.to_csv('evaluation_metrics.csv', index=False)
        return df

def main():
    # Initialize detector
    detector = StudentBehaviorDetector('C:/Users/enock/Downloads/Project/students-behaviors-detection.v19i.yolov11/data.yaml')
    
    # Train model
    detector.train_model(epochs=100)
    
    # Evaluate and generate visualizations
    val_results = detector.evaluate_model()
    detector.plot_training_metrics()
    detector.plot_confusion_matrix('C:/Users/enock/Downloads/Project/students-behaviors-detection.v19i.yolov11/valid/labels')
    detector.plot_class_distribution()
    
    # Export metrics report
    metrics_df = detector.export_metrics_report(val_results)
    print("\nEvaluation Metrics Summary:")
    print(metrics_df)

if __name__ == "__main__":
    main()
