#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run the full oral cancer detection pipeline')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to data directory containing Images, Annotation.json, etc.')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Path to output directory for processed data')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training and inference')
    parser.add_argument('--device', type=str, default='', help='Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--model', type=str, default='yolov8n', help='YOLO model to use (yolov8n, yolov8s, etc.)')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip data analysis step')
    parser.add_argument('--skip-prep', action='store_true', help='Skip data preparation steps')
    parser.add_argument('--skip-train', action='store_true', help='Skip training step')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation step')
    
    return parser.parse_args()

def run_command(command, description):
    """Run a command and print its output"""
    print(f"\n{'-' * 80}")
    print(f"STEP: {description}")
    print(f"{'-' * 80}")
    
    start_time = time.time()
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    end_time = time.time()
    
    print(f"\nCommand completed in {end_time - start_time:.2f} seconds with exit code {process.returncode}")
    
    if process.returncode != 0:
        print(f"WARNING: Command exited with non-zero status: {process.returncode}")
    
    return process.returncode

def main(args):
    # Define paths
    data_dir = args.data_dir
    output_dir = args.output_dir
    
    # Get the path to the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    image_dir = os.path.join(data_dir, "Images")
    annotation_file = os.path.join(data_dir, "Annotation.json")
    imagewise_csv = os.path.join(data_dir, "Imagewise_Data.csv")
    patientwise_csv = os.path.join(data_dir, "Patientwise_Data.csv")
    
    # Step 1: Analyze data (optional)
    if not args.skip_analysis:
        print("\nStep 1: Analyzing data...")
        if not os.path.exists("results"):
            os.makedirs("results", exist_ok=True)
        
        run_command(
            f"python {os.path.join(script_dir, 'utils/data_analysis.py')} --patient_csv '{patientwise_csv}' --image_csv '{imagewise_csv}' --coco_file '{annotation_file}'",
            "Analyzing data"
        )
    else:
        print("\nSkipping data analysis step...")
    
    if not args.skip_prep:
        # Step 2: Convert COCO annotations to YOLO format
        print("\nStep 2: Converting COCO annotations to YOLO format...")
        yolo_labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(yolo_labels_dir, exist_ok=True)
        
        run_command(
            f"python {os.path.join(script_dir, 'utils/coco_to_yolo.py')} --coco_file '{annotation_file}' --output_dir '{yolo_labels_dir}' --image_dir '{image_dir}'",
            "Converting COCO annotations to YOLO format"
        )
        
        # Step 3: Prepare dataset (split into train/val/test)
        print("\nStep 3: Preparing dataset...")
        run_command(
            f"python {os.path.join(script_dir, 'utils/prepare_dataset.py')} --imagewise_csv '{imagewise_csv}' --patient_wise_csv '{patientwise_csv}' "
            f"--image_dir '{image_dir}' --output_dir '{output_dir}'",
            "Preparing dataset"
        )
    
    # Step 4: Train model
    if not args.skip_train:
        print("\nStep 4: Training model...")
        # Ensure we have absolute path to dataset.yaml
        dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
        if not os.path.isabs(dataset_yaml):
            dataset_yaml = os.path.abspath(dataset_yaml)
        
        run_command(
            f"python {os.path.join(script_dir, 'train.py')} --config {os.path.join(script_dir, 'configs/yolov11.yaml')} "
            f"--data '{dataset_yaml}' --model {args.model} "
            f"--epochs {args.epochs} --batch-size {args.batch_size} --img-size {args.img_size} --device {args.device}",
            f"Training {args.model} model"
        )
    
    # Step 5: Evaluate model
    if not args.skip_eval:
        print("\nStep 5: Evaluating model...")
        best_model = os.path.join("runs/oral_cancer/yolov11_detection/weights/best.pt")
        
        if not os.path.exists(best_model):
            print(f"Warning: Best model not found at {best_model}. Using last.pt instead.")
            best_model = os.path.join("runs/oral_cancer/yolov11_detection/weights/last.pt")
        
        if os.path.exists(best_model):
            # Ensure we have absolute path to dataset.yaml
            dataset_yaml = os.path.join(output_dir, 'dataset.yaml')
            if not os.path.isabs(dataset_yaml):
                dataset_yaml = os.path.abspath(dataset_yaml)
                
            run_command(
                f"python {os.path.join(script_dir, 'evaluate.py')} --model '{best_model}' --data '{dataset_yaml}' "
                f"--batch-size {args.batch_size} --img-size {args.img_size} --device {args.device}",
                "Evaluating YOLO model"
            )
        else:
            print("Error: No trained model found. Skipping evaluation.")
    
    print("\nPipeline completed successfully!")
    print(f"Processed data: {output_dir}")
    print(f"Training logs: runs/oral_cancer/yolov11_detection")
    print(f"Evaluation results: results/evaluation")

if __name__ == '__main__':
    args = parse_args()
    main(args) 