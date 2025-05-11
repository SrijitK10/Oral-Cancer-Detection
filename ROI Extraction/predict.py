#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with YOLOv11 for oral cancer detection')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv11 model')
    parser.add_argument('--img', type=str, help='Path to image file')
    parser.add_argument('--dir', type=str, help='Path to directory of images')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='', help='Device to run inference on (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--save-dir', type=str, default='results/predictions', help='Directory to save results')
    parser.add_argument('--view', action='store_true', help='Display results')
    parser.add_argument('--class-names', type=str, help='Path to YAML file with class names')
    
    return parser.parse_args()

def predict(model_path, img_path=None, dir_path=None, conf=0.25, iou=0.45, 
            device='', save_dir='results/predictions', view=False, class_names_path=None):
    """
    Run inference with YOLOv11 model
    
    Args:
        model_path (str): Path to YOLOv11 model
        img_path (str, optional): Path to image file
        dir_path (str, optional): Path to directory of images
        conf (float): Confidence threshold
        iou (float): NMS IoU threshold
        device (str): Device to run inference on
        save_dir (str): Directory to save results
        view (bool): Display results
        class_names_path (str, optional): Path to YAML file with class names
    """
    # Load model
    model = YOLO(model_path)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Get class names
    if class_names_path:
        with open(class_names_path, 'r') as f:
            config = yaml.safe_load(f)
            class_names = config.get('names', [])
    else:
        try:
            # Try to get class names from model
            class_names = model.names
        except AttributeError:
            class_names = ['Oral Cavity', 'Healthy', 'Benign', 'OPMD', 'OCA']
    
    # Set up custom colors for classes
    colors = {
        0: (0, 255, 0),    # Oral Cavity - Green
        1: (255, 255, 255), # Healthy - White
        2: (255, 255, 0),   # Benign - Yellow
        3: (255, 165, 0),   # OPMD - Orange
        4: (255, 0, 0)      # OCA - Red
    }
    
    # Run inference on single image
    if img_path:
        results = model.predict(
            source=img_path,
            conf=conf,
            iou=iou,
            device=device,
            save=True,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            show=view,
            project=save_dir,
            name='single_image'
        )
        
        # Access the results
        for r in results:
            image_path = r.path
            boxes = r.boxes  # Boxes object for bounding box outputs
            filename = Path(image_path).stem
            
            # Draw boxes with custom colors and save
            orig_img = cv2.imread(image_path)
            for box in boxes:
                # Get box coordinates and class
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Draw box with class color
                color = colors.get(cls, (0, 0, 255))  # Default to red if class not found
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
                
                # Label with class name and confidence
                label = f"{class_names[cls]} {conf:.2f}"
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(orig_img, (x1, y1), c2, color, -1)
                cv2.putText(orig_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            # Save annotated image
            custom_path = os.path.join(save_dir, 'single_image', f"{filename}_annotated.jpg")
            cv2.imwrite(custom_path, orig_img)
            print(f"Saved annotated image to {custom_path}")
    
    # Run inference on directory of images
    if dir_path:
        results = model.predict(
            source=dir_path,
            conf=conf,
            iou=iou,
            device=device,
            save=True,
            save_txt=True,
            save_conf=True,
            save_crop=False,
            show=view,
            project=save_dir,
            name='batch'
        )
        print(f"Processed {len(results)} images. Results saved to {os.path.join(save_dir, 'batch')}")
    
    return results

if __name__ == '__main__':
    args = parse_args()
    
    if not args.img and not args.dir:
        print("Error: You must provide either --img or --dir")
        exit(1)
    
    results = predict(
        args.model,
        args.img,
        args.dir,
        args.conf,
        args.iou,
        args.device,
        args.save_dir,
        args.view,
        args.class_names
    ) 