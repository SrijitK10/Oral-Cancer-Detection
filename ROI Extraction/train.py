#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO for oral cancer detection')
    parser.add_argument('--config', type=str, default='configs/yolov11.yaml', help='Path to config file')
    parser.add_argument('--data', type=str, help='Path to dataset.yaml file')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--img-size', type=int, help='Image size')
    parser.add_argument('--device', type=str, help='Device to train on (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--weights', type=str, help='Initial weights path')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--model', type=str, default='yolov8n', help='Model type (yolov8n, yolov8s, etc.)')
    
    return parser.parse_args()

def find_dataset_file(data_path):
    """
    Find the dataset file by resolving absolute and relative paths.
    
    Args:
        data_path (str): Path to the dataset YAML file
        
    Returns:
        str: Resolved path to the dataset file
    """
    # If it's already an absolute path and exists, return it
    if os.path.isabs(data_path) and os.path.exists(data_path):
        return data_path
    
    # Try the path as provided
    if os.path.exists(data_path):
        return os.path.abspath(data_path)
    
    # Try with parent directory
    parent_path = os.path.join('..', data_path)
    if os.path.exists(parent_path):
        return os.path.abspath(parent_path)
    
    # Try with full parent path
    full_parent_path = os.path.abspath(parent_path)
    if os.path.exists(full_parent_path):
        return full_parent_path
    
    # If we have a path like '../data/processed/dataset.yaml'
    # Try to resolve it from the current directory
    base_path = os.path.abspath('.')
    while '..' in data_path:
        data_path = data_path.replace('..', '')
        data_path = data_path.lstrip('/')
        potential_path = os.path.join(os.path.dirname(base_path), data_path)
        if os.path.exists(potential_path):
            return potential_path
    
    # If all else fails, return the original path and let the caller handle the error
    return data_path

def download_yolo_model(model_name="yolov8n"):
    """Download YOLO weights if they don't exist"""
    import os
    
    # Directory to save weights
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the model to trigger download if needed
    print(f"Loading {model_name} model...")
    model = YOLO(model_name)
    
    print(f"Model {model_name} loaded successfully")
    return model_name

def train(config_path, model_name="yolov8n", **kwargs):
    """
    Train YOLO model
    
    Args:
        config_path (str): Path to config file
        model_name (str): Name of the model to use (yolov8n, yolov8s, etc.)
        **kwargs: Override config parameters
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with kwargs
    for k, v in kwargs.items():
        if v is not None:
            # Handle nested config
            if '.' in k:
                parts = k.split('.')
                c = config
                for p in parts[:-1]:
                    if p not in c:
                        c[p] = {}
                    c = c[p]
                c[parts[-1]] = v
            else:
                config[k] = v
    
    # Find and resolve dataset path if it exists in config
    if 'data' in kwargs:
        resolved_data_path = find_dataset_file(kwargs['data'])
        print(f"Using dataset at: {resolved_data_path}")
        kwargs['data'] = resolved_data_path
    
    # Ensure YOLO model is available
    model_name = download_yolo_model(model_name)
    
    # Initialize YOLO model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=config['data'] if 'data' in kwargs else config['path'],
        epochs=config['train']['epochs'],
        imgsz=config['train']['imgsz'],
        batch=config['train']['batch'],
        device=config['train']['device'],
        workers=config['train']['workers'],
        optimizer=config['train']['optimizer'],
        lr0=config['train']['lr0'],
        lrf=config['train']['lrf'],
        momentum=config['train']['momentum'],
        weight_decay=config['train']['weight_decay'],
        warmup_epochs=config['train']['warmup_epochs'],
        warmup_momentum=config['train']['warmup_momentum'],
        warmup_bias_lr=config['train']['warmup_bias_lr'],
        box=config['train']['box'],
        cls=config['train']['cls'],
        dfl=config['train']['dfl'],
        close_mosaic=config['train']['close_mosaic'],
        patience=config['train']['patience'],
        save_period=config['train']['save_period'],
        freeze=config['train']['freeze'],
        project=config['project'],
        name=config['name'],
        exist_ok=config['exist_ok'],
        resume=config['train']['resume'] if 'resume' not in kwargs else kwargs['resume'],
        pretrained=config['model']['pretrained'],
    )
    
    return results

if __name__ == '__main__':
    args = parse_args()
    
    # Prepare kwargs to override config
    kwargs = {
        'data': args.data,
        'train.epochs': args.epochs,
        'train.batch': args.batch_size,
        'train.imgsz': args.img_size,
        'train.device': args.device,
        'model.weights': args.weights,
        'resume': args.resume,
    }
    
    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    # Train the model
    results = train(args.config, args.model, **kwargs)
    
    print(f"Training complete. Results saved to {results}") 