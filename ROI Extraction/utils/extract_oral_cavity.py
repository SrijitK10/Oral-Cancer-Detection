#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import cv2
from PIL import Image
import numpy as np

def extract_oral_cavity(coco_file, image_dir, output_dir, oral_cavity_class_id=2):
    """
    Extract oral cavity regions from images using COCO annotations
    
    Args:
        coco_file (str): Path to COCO annotation file
        image_dir (str): Directory containing the images
        output_dir (str): Directory to save extracted oral cavity images
        oral_cavity_class_id (int): Class ID for oral cavity in annotations (default: 2)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load COCO annotations
    print("Loading COCO annotations...")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Print category information
    print("\nCategory Information:")
    categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
    for cat_id, cat_info in categories.items():
        print(f"  Category ID {cat_id}: {cat_info.get('name')} (Supercategory: {cat_info.get('supercategory')})")
    
    # Check if the specified class ID exists
    if oral_cavity_class_id not in categories:
        print(f"Error: Category ID {oral_cavity_class_id} not found in annotations.")
        # Create a simple category list string without using nested f-strings
        category_list = []
        for cat_id, cat_info in categories.items():
            category_list.append(f"{cat_id}: {cat_info.get('name')}")
        print(f"Available categories: {', '.join(category_list)}")
        return
    
    print(f"\nExtracting regions for category: {categories[oral_cavity_class_id]['name']}")
    
    # Get image info
    images = {img.get('id'): img for img in coco_data.get('images', [])}
    
    # Process annotations for oral cavity class
    print("Processing annotations...")
    
    # Group annotations by image ID
    annotations_by_image = {}
    total_annotations = 0
    for ann in coco_data.get('annotations', []):
        if ann.get('category_id') == oral_cavity_class_id:
            img_id = ann.get('image_id')
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            total_annotations += 1
    
    print(f"Found {total_annotations} annotations for category ID {oral_cavity_class_id} across {len(annotations_by_image)} images.")
    
    if total_annotations == 0:
        print("No annotations found for the specified category ID. Please check if the category ID is correct.")
        return
    
    # Process each image
    print(f"Extracting regions from {len(annotations_by_image)} images...")
    
    extracted_count = 0
    for img_id, anns in tqdm(annotations_by_image.items()):
        if img_id not in images:
            continue
            
        img_info = images[img_id]
        img_filename = img_info.get('file_name')
        
        # Get image file path
        img_path = os.path.join(image_dir, img_filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found. Skipping.")
            continue
        
        try:
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image {img_path}. Skipping.")
                continue
                
            # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process each annotation for this image
            for i, ann in enumerate(anns):
                bbox = ann.get('bbox', [0, 0, 0, 0])
                x_min, y_min, width, height = [int(coord) for coord in bbox]
                
                # Extract the region
                # Ensure boundaries are within image dimensions
                img_height, img_width = img_rgb.shape[:2]
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img_width, x_min + width)
                y_max = min(img_height, y_min + height)
                
                # Skip if bounding box is invalid
                if x_max <= x_min or y_max <= y_min:
                    print(f"Warning: Invalid bounding box in {img_filename}. Skipping.")
                    continue
                
                # Crop the region
                cropped_region = img_rgb[y_min:y_max, x_min:x_max]
                
                # Save the cropped image with the same filename
                if len(anns) > 1:
                    # If multiple regions in one image, add index
                    output_filename = f"{img_filename.split('.')[0]}_{i}.{img_filename.split('.')[-1]}"
                else:
                    output_filename = img_filename
                
                output_path = os.path.join(output_dir, output_filename)
                
                # Save using PIL to ensure proper color handling
                Image.fromarray(cropped_region).save(output_path)
                extracted_count += 1
                
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
    
    print(f"Extraction complete. {extracted_count} regions saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Extract regions from images using COCO annotations')
    parser.add_argument('--coco_file', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing the images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save extracted regions')
    parser.add_argument('--class_id', type=int, default=2, help='Class ID for the region to extract (default: 2 for Oral Cavity)')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    extract_oral_cavity(args.coco_file, args.image_dir, args.output_dir, args.class_id)