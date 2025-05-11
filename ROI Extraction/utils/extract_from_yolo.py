#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import yaml

def extract_from_yolo(data_dir, output_dir, oral_cavity_class=2):
    """
    Extract oral cavity regions using YOLO format annotations, preserving train/val/test splits
    
    Args:
        data_dir (str): Path to the processed data directory (containing train, val, test folders)
        output_dir (str): Directory to save extracted oral cavity images
        oral_cavity_class (int): Class ID for oral cavity in YOLO annotations (default: 2)
    """
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Load dataset configuration
    dataset_yaml = os.path.join(data_dir, 'dataset.yaml')
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    # Print class names for reference
    print(f"Class names in dataset: {dataset_config.get('names', [])}")
    
    # Check if the specified class exists
    class_names = dataset_config.get('names', [])
    if oral_cavity_class >= len(class_names):
        print(f"Error: Class ID {oral_cavity_class} is out of range. Available classes: {class_names}")
        return
    
    print(f"Extracting regions for class: {class_names[oral_cavity_class]}")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        labels_dir = os.path.join(data_dir, split, 'labels')
        images_dir = os.path.join(data_dir, split, 'images')
        
        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"Warning: {labels_dir} or {images_dir} not found. Skipping.")
            continue
        
        # Get all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        print(f"Found {len(label_files)} label files")
        
        extracted_count = 0
        for label_file in tqdm(label_files):
            base_name = os.path.splitext(label_file)[0]
            image_file = base_name + '.jpg'
            image_path = os.path.join(images_dir, image_file)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found. Skipping.")
                continue
            
            # Read label file
            bboxes = []
            try:
                with open(os.path.join(labels_dir, label_file), 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        class_id = int(parts[0])
                        
                        # Only process bounding boxes for the oral cavity class
                        if class_id == oral_cavity_class:
                            # YOLO format: class_id, x_center, y_center, width, height (normalized)
                            x_center, y_center, width, height = map(float, parts[1:5])
                            bboxes.append((x_center, y_center, width, height))
            except Exception as e:
                print(f"Error reading label file {label_file}: {e}")
                continue
            
            if not bboxes:
                # No oral cavity annotations in this image
                continue
            
            try:
                # Read the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Error reading image {image_path}. Skipping.")
                    continue
                
                # Convert from BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_height, img_width = img_rgb.shape[:2]
                
                # Process each bounding box
                for i, (x_center, y_center, width, height) in enumerate(bboxes):
                    # Convert normalized YOLO coordinates to absolute pixel coordinates
                    x_min = int((x_center - width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    x_max = int((x_center + width / 2) * img_width)
                    y_max = int((y_center + height / 2) * img_height)
                    
                    # Ensure boundaries are within image dimensions
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(img_width, x_max)
                    y_max = min(img_height, y_max)
                    
                    # Skip if bounding box is invalid
                    if x_max <= x_min or y_max <= y_min:
                        print(f"Warning: Invalid bounding box in {image_file}. Skipping.")
                        continue
                    
                    # Crop the region
                    cropped_region = img_rgb[y_min:y_max, x_min:x_max]
                    
                    # Save the cropped image with the same filename
                    if len(bboxes) > 1:
                        output_filename = f"{base_name}_{i}.jpg"
                    else:
                        output_filename = image_file
                    
                    output_path = os.path.join(output_dir, split, output_filename)
                    
                    # Save using PIL to ensure proper color handling
                    Image.fromarray(cropped_region).save(output_path)
                    extracted_count += 1
                    
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        print(f"Extracted {extracted_count} {class_names[oral_cavity_class]} regions in {split} split")
    
    print("\nExtraction complete!")

def parse_args():
    parser = argparse.ArgumentParser(description='Extract oral cavity regions from YOLO format annotations')
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='data/oral_cavity_crops', help='Directory to save extracted regions')
    parser.add_argument('--class_id', type=int, default=2, help='Class ID for the region to extract (default: 2 for Oral Cavity)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_from_yolo(args.data_dir, args.output_dir, args.class_id) 