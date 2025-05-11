#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 model for oral cancer detection')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv11 model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device to evaluate on (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--save-dir', type=str, default='results/evaluation', help='Directory to save evaluation results')
    
    return parser.parse_args()

def evaluate(model_path, data_path, conf, iou, device, batch_size, img_size, save_dir):
    """
    Evaluate YOLOv11 model
    
    Args:
        model_path (str): Path to YOLOv11 model
        data_path (str): Path to dataset.yaml file
        conf (float): Confidence threshold
        iou (float): NMS IoU threshold
        device (str): Device to evaluate on
        batch_size (int): Batch size
        img_size (int): Image size
        save_dir (str): Directory to save evaluation results
    """
    # Load model
    model = YOLO(model_path)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Evaluate model on validation set
    print(f"Evaluating model on validation set...")
    val_results = model.val(
        data=data_path,
        split='val',
        conf=conf,
        iou=iou,
        device=device,
        batch=batch_size,
        imgsz=img_size,
        save_json=True,
        save_dir=save_dir
    )
    
    # Evaluate model on test set
    print(f"Evaluating model on test set...")
    test_results = model.val(
        data=data_path,
        split='test',
        conf=conf,
        iou=iou,
        device=device,
        batch=batch_size,
        imgsz=img_size,
        save_json=True,
        save_dir=save_dir
    )
    
    # Load class names from dataset.yaml
    with open(data_path, 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config.get('names', [])
    
    # Generate confusion matrix
    print(f"Generating confusion matrix...")
    # Note: This is a simplified example. In practice, you'd need to collect
    # predictions and ground truth for each image and create a confusion matrix.
    
    # Save summary of results
    summary = {
        'Model': model_path,
        'Dataset': data_path,
        'Image Size': img_size,
        'Confidence Threshold': conf,
        'IoU Threshold': iou,
        'Validation mAP50': val_results.box.map50,
        'Validation mAP50-95': val_results.box.map,
        'Test mAP50': test_results.box.map50,
        'Test mAP50-95': test_results.box.map,
    }
    
    with open(f"{save_dir}/summary.yaml", 'w') as f:
        yaml.dump(summary, f)
    
    # Plot evaluation metrics
    metrics = {
        'mAP@0.5': [val_results.box.map50, test_results.box.map50],
        'mAP@0.5:0.95': [val_results.box.map, test_results.box.map],
        'Precision': [val_results.box.mp, test_results.box.mp],
        'Recall': [val_results.box.mr, test_results.box.mr]
    }
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame(metrics, index=['Validation', 'Test'])
    ax = df.plot(kind='bar', rot=0, width=0.7)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', fontsize=10)
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"  Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"  Validation mAP@0.5:0.95: {val_results.box.map:.4f}")
    print(f"  Test mAP@0.5: {test_results.box.map50:.4f}")
    print(f"  Test mAP@0.5:0.95: {test_results.box.map:.4f}")
    print(f"\nResults saved to {save_dir}")
    
    return val_results, test_results

if __name__ == '__main__':
    args = parse_args()
    
    val_results, test_results = evaluate(
        args.model,
        args.data,
        args.conf,
        args.iou,
        args.device,
        args.batch_size,
        args.img_size,
        args.save_dir
    ) 