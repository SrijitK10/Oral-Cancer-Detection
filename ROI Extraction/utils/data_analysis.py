#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path

def analyze_patient_data(patient_csv):
    """
    Analyze patient-level data
    
    Args:
        patient_csv (str): Path to patient-wise data CSV
    """
    # Load patient data
    df = pd.read_csv(patient_csv)
    
    # Clean column names - remove trailing spaces
    df.columns = df.columns.str.strip()
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('results/age_distribution.png', dpi=300, bbox_inches='tight')
    
    # Gender distribution
    plt.figure(figsize=(8, 6))
    gender_counts = df['Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Gender Distribution')
    plt.savefig('results/gender_distribution.png', dpi=300, bbox_inches='tight')
    
    # Risk factors
    risk_factors = ['Smoking', 'Chewing_Betel_Quid', 'Alcohol']
    plt.figure(figsize=(12, 6))
    
    for i, factor in enumerate(risk_factors):
        plt.subplot(1, 3, i+1)
        counts = df[factor].value_counts()
        plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
        plt.title(factor.replace('_', ' '))
    
    plt.tight_layout()
    plt.savefig('results/risk_factors.png', dpi=300, bbox_inches='tight')
    
    # Risk factor combinations
    plt.figure(figsize=(10, 6))
    # Create risk factor combination column
    df['risk_combination'] = df.apply(
        lambda row: f"Smoking: {row['Smoking']}, Betel: {row['Chewing_Betel_Quid']}, Alcohol: {row['Alcohol']}", 
        axis=1
    )
    
    # Count combinations
    risk_combos = df['risk_combination'].value_counts()
    
    # Plot top combinations
    top_combos = risk_combos.head(6)
    sns.barplot(x=top_combos.index, y=top_combos.values)
    plt.title('Top Risk Factor Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/risk_combinations.png', dpi=300, bbox_inches='tight')
    
    print(f"Patient data analysis complete. Results saved to 'results/' directory.")

def analyze_image_data(image_csv):
    """
    Analyze image-level data
    
    Args:
        image_csv (str): Path to image-wise data CSV
    """
    # Load image data
    df = pd.read_csv(image_csv)
    
    # Category distribution
    plt.figure(figsize=(10, 6))
    category_counts = df['Category'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values)
    plt.title('Distribution of Image Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/category_distribution.png', dpi=300, bbox_inches='tight')
    
    # Clinical diagnosis distribution
    plt.figure(figsize=(14, 8))
    diagnosis_counts = df['Clinical Diagnosis'].value_counts().head(15)  # Top 15 diagnoses
    sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values)
    plt.title('Top 15 Clinical Diagnoses')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/diagnosis_distribution.png', dpi=300, bbox_inches='tight')
    
    # Lesion annotation count distribution
    plt.figure(figsize=(10, 6))
    lesion_counts = df['Lesion Annotation Count'].value_counts().sort_index()
    sns.barplot(x=lesion_counts.index, y=lesion_counts.values)
    plt.title('Distribution of Lesion Annotation Counts')
    plt.xlabel('Number of Lesion Annotations')
    plt.ylabel('Count of Images')
    plt.tight_layout()
    plt.savefig('results/lesion_count_distribution.png', dpi=300, bbox_inches='tight')
    
    # Category vs. Lesion Annotation Count
    plt.figure(figsize=(12, 6))
    category_lesion = df.groupby('Category')['Lesion Annotation Count'].mean().reset_index()
    sns.barplot(x='Category', y='Lesion Annotation Count', data=category_lesion)
    plt.title('Average Lesion Annotation Count by Category')
    plt.ylabel('Average Count')
    plt.tight_layout()
    plt.savefig('results/category_vs_lesion_count.png', dpi=300, bbox_inches='tight')
    
    print(f"Image data analysis complete. Results saved to 'results/' directory.")

def analyze_coco_annotations(coco_file):
    """
    Analyze COCO annotations
    
    Args:
        coco_file (str): Path to COCO annotation file
    """
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Extract basic stats
    num_images = len(coco_data.get('images', []))
    num_annotations = len(coco_data.get('annotations', []))
    num_categories = len(coco_data.get('categories', []))
    
    print(f"\nCOCO Annotation Stats:")
    print(f"  Number of images: {num_images}")
    print(f"  Number of annotations: {num_annotations}")
    print(f"  Number of categories: {num_categories}")
    
    # Distribution of annotations per image
    annotations_per_image = Counter()
    for ann in coco_data.get('annotations', []):
        annotations_per_image[ann['image_id']] += 1
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    ann_counts = list(annotations_per_image.values())
    sns.histplot(ann_counts, bins=max(10, min(50, len(set(ann_counts)))), kde=True)
    plt.title('Distribution of Annotations per Image')
    plt.xlabel('Number of Annotations')
    plt.ylabel('Count of Images')
    plt.tight_layout()
    plt.savefig('results/annotations_per_image.png', dpi=300, bbox_inches='tight')
    
    # Annotations per category
    category_counts = Counter()
    for ann in coco_data.get('annotations', []):
        category_counts[ann['category_id']] += 1
    
    # Map category IDs to names
    category_map = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Create category name counts
    category_name_counts = {category_map.get(cat_id, f"Unknown ({cat_id})"): count 
                           for cat_id, count in category_counts.items()}
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(category_name_counts.keys(), category_name_counts.values())
    plt.title('Distribution of Annotations by Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/annotations_by_category.png', dpi=300, bbox_inches='tight')
    
    # Analyze bounding box sizes
    areas = []
    aspect_ratios = []
    
    for ann in coco_data.get('annotations', []):
        if 'bbox' in ann:
            x, y, width, height = ann['bbox']
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            areas.append(area)
            aspect_ratios.append(aspect_ratio)
    
    # Plot area distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(areas, bins=50, kde=True)
    plt.title('Distribution of Bounding Box Areas')
    plt.xlabel('Area (pixels²)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/bbox_area_distribution.png', dpi=300, bbox_inches='tight')
    
    # Plot aspect ratio distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(aspect_ratios, bins=50, kde=True)
    plt.title('Distribution of Bounding Box Aspect Ratios (width/height)')
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('results/bbox_aspect_ratio_distribution.png', dpi=300, bbox_inches='tight')
    
    print(f"COCO annotation analysis complete. Results saved to 'results/' directory.")

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze oral cancer dataset')
    parser.add_argument('--patient_csv', type=str, help='Path to patient-wise data CSV')
    parser.add_argument('--image_csv', type=str, help='Path to image-wise data CSV')
    parser.add_argument('--coco_file', type=str, help='Path to COCO annotation file')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    if args.patient_csv:
        analyze_patient_data(args.patient_csv)
    
    if args.image_csv:
        analyze_image_data(args.image_csv)
    
    if args.coco_file:
        analyze_coco_annotations(args.coco_file) 