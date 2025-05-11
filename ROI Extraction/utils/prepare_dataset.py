#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import csv

def create_splits(imagewise_csv, patient_wise_csv, image_dir, output_dir, 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train, validation, and test splits for the dataset
    
    Args:
        imagewise_csv (str): Path to image-wise data CSV
        patient_wise_csv (str): Path to patient-wise data CSV
        image_dir (str): Directory containing the images
        output_dir (str): Directory to save the splits
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
    """
    # Load image-wise data
    image_data = pd.read_csv(imagewise_csv)
    patient_data = pd.read_csv(patient_wise_csv)
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    # Get patient IDs
    patient_ids = list(patient_data['Patient ID'].unique())
    
    # Split patient IDs to ensure patients don't overlap between splits
    train_patients, temp_patients = train_test_split(
        patient_ids, test_size=(val_ratio + test_ratio), random_state=42
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=test_ratio/(val_ratio + test_ratio), random_state=42
    )
    
    # Create a mapping from patient ID to split
    patient_to_split = {}
    for patient_id in train_patients:
        patient_to_split[patient_id] = 'train'
    for patient_id in val_patients:
        patient_to_split[patient_id] = 'val'
    for patient_id in test_patients:
        patient_to_split[patient_id] = 'test'
    
    # Create split for each image
    image_to_split = {}
    for _, row in image_data.iterrows():
        image_name = row['Image Name']
        patient_id = image_name.split('-')[0] + '-' + image_name.split('-')[1]
        
        if patient_id in patient_to_split:
            image_to_split[image_name] = patient_to_split[patient_id]
    
    # Copy images and labels to respective splits
    print("Copying files to train, validation, and test splits...")
    for image_name, split in tqdm(image_to_split.items()):
        # Source image and label paths
        image_path = os.path.join(image_dir, f"{image_name}.jpg")
        label_path = os.path.join(output_dir, "labels", f"{image_name}.txt")
        
        # Destination paths
        dest_image_path = os.path.join(output_dir, split, "images", f"{image_name}.jpg")
        dest_label_path = os.path.join(output_dir, split, "labels", f"{image_name}.txt")
        
        # Copy if source files exist
        if os.path.exists(image_path) and os.path.exists(label_path):
            shutil.copy(image_path, dest_image_path)
            shutil.copy(label_path, dest_label_path)
    
    # Convert output_dir to absolute path if it's not already
    abs_output_dir = os.path.abspath(output_dir)
    
    # Create dataset.yaml for YOLO
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO dataset configuration\n")
        f.write(f"path: {abs_output_dir}\n")
        f.write(f"train: {os.path.join(abs_output_dir, 'train/images')}\n")
        f.write(f"val: {os.path.join(abs_output_dir, 'val/images')}\n")
        f.write(f"test: {os.path.join(abs_output_dir, 'test/images')}\n\n")
        
        # Classes
        f.write(f"nc: 5  # number of classes\n")
        f.write(f"names: ['Oral Cavity', 'Healthy', 'Benign', 'OPMD', 'OCA']\n")
    
    # Create statistics file
    stats = {
        'train': len([img for img, split in image_to_split.items() if split == 'train']),
        'val': len([img for img, split in image_to_split.items() if split == 'val']),
        'test': len([img for img, split in image_to_split.items() if split == 'test']),
    }
    
    print(f"Dataset split complete:")
    print(f"  Train: {stats['train']} images")
    print(f"  Validation: {stats['val']} images")
    print(f"  Test: {stats['test']} images")
    
    # Create class distribution statistics
    class_stats = {}
    for _, row in image_data.iterrows():
        image_name = row['Image Name']
        category = row['Category']
        
        if image_name in image_to_split:
            split = image_to_split[image_name]
            
            if category not in class_stats:
                class_stats[category] = {'train': 0, 'val': 0, 'test': 0}
            
            class_stats[category][split] += 1
    
    # Write class distribution to CSV
    with open(os.path.join(output_dir, 'class_distribution.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'Train', 'Validation', 'Test', 'Total'])
        
        for category, counts in class_stats.items():
            writer.writerow([
                category, 
                counts['train'], 
                counts['val'], 
                counts['test'],
                counts['train'] + counts['val'] + counts['test']
            ])
    
    print(f"Dataset prepared successfully. Configuration saved to {yaml_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLOv11 training')
    parser.add_argument('--imagewise_csv', type=str, required=True, help='Path to image-wise data CSV')
    parser.add_argument('--patient_wise_csv', type=str, required=True, help='Path to patient-wise data CSV')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the splits')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test data')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_splits(args.imagewise_csv, args.patient_wise_csv, args.image_dir, args.output_dir, 
                 args.train_ratio, args.val_ratio, args.test_ratio) 