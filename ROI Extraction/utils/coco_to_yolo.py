#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

def coco_to_yolo(coco_file, output_dir, image_dir):
    """
    Convert COCO format annotations to YOLO format
    
    Args:
        coco_file (str): Path to COCO annotation file
        output_dir (str): Directory to save YOLO format annotations
        image_dir (str): Directory containing the images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get image info
    images = {img.get('id'): img for img in coco_data.get('images', [])}
    
    # Get category info
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # Process annotations
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann.get('image_id')
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Create class mapping
    # In this case, we have 4 categories: Healthy, Benign, OPMD, OCA
    # Class IDs should match with those in the COCO annotations
    # 0: Oral cavity (region of interest)
    # 1: Healthy lesion
    # 2: Benign lesion
    # 3: OPMD lesion
    # 4: OCA lesion
    
    # Process each image
    print(f"Converting annotations to YOLO format...")
    
    for img_id, img_info in tqdm(images.items()):
        img_filename = img_info.get('file_name')
        
        # Skip if image has no annotations
        if img_id not in annotations_by_image:
            continue
        
        # Get image dimensions directly from the image file
        img_path = os.path.join(image_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
        
        try:
            # Get image dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}. Skipping.")
            continue
        
        # Get annotations for this image
        img_annotations = annotations_by_image[img_id]
        
        # Create YOLO annotation file
        txt_filename = os.path.splitext(img_filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for ann in img_annotations:
                # Get category id
                category_id = ann.get('category_id', 0)
                
                # Convert bbox to YOLO format
                # COCO: [x_min, y_min, width, height]
                # YOLO: [x_center, y_center, width, height] (normalized)
                bbox = ann.get('bbox', [0, 0, 0, 0])
                x_min, y_min, width, height = bbox
                
                # Convert to YOLO format (normalized)
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # Write to file
                yolo_bbox = [x_center, y_center, norm_width, norm_height]
                f.write(f"{category_id} " + " ".join([str(round(coord, 6)) for coord in yolo_bbox]) + "\n")
    
    print(f"Conversion complete. YOLO format annotations saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO format annotations to YOLO format')
    parser.add_argument('--coco_file', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save YOLO format annotations')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    coco_to_yolo(args.coco_file, args.output_dir, args.image_dir) 