"""
Main training pipeline for the Oral Cancer Classification Project.

This script orchestrates the complete training pipeline:
- Loading and parsing configuration
- Data preparation and augmentation
- Model training (baseline, transfer learning models, ensemble)
- Evaluation and visualization

Example usage:
    python train.py --config ./config/config.yml
    python train.py --data_dir /path/to/data --batch_size 16 --epochs 50
    python train.py --no_augment
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import project modules
from src.utils.config_utils import load_config, parse_arguments, merge_configs
from src.data.data_loader import (
    analyze_dataset, 
    generate_augmented_images, 
    create_data_generators
)
from src.models.models import (
    create_baseline_model,
    create_transfer_learning_model, 
    train_model,
    create_ensemble_model
)
from src.models.evaluation import evaluate_models
from src.visualization.visualization import (
    plot_class_distribution,
    plot_training_history,
    plot_model_comparison
)


def main():
    """Main entry point for the training pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Merge configuration and command-line arguments
    config = merge_configs(config, args)
    
    # Create output directories
    os.makedirs(config["models"]["save_dir"], exist_ok=True)
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    
    print("\n===== Oral Cancer Classification Pipeline =====")
    print(f"Mode: {args.mode}")
    print(f"Configuration: {args.config}")
    print(f"Dataset: {config['dataset']['base_dir']}")
    
    # Analyze dataset
    train_counts, test_counts, val_counts, max_count = analyze_dataset(config)
    
    # Generate augmented images if enabled
    use_augmented = config["augmentation"].get("enabled", True)
    
    if use_augmented:
        augmented_dir = generate_augmented_images(config, train_counts, max_count)
        augmented_counts = analyze_dataset(config)[0]  # Get updated counts
        
        # Plot class distribution before and after augmentation
        plot_class_distribution(
            train_counts, 
            augmented_counts,
            save_path=os.path.join(config["output"]["results_dir"], "class_distribution.png")
        )
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators(config, use_augmented=use_augmented)
    
    print(f"\nTraining set: {train_gen.samples} images")
    print(f"Validation set: {val_gen.samples} images")
    print(f"Test set: {test_gen.samples} images")
    print(f"Number of classes: {len(train_gen.class_indices)}")
    print(f"Class mapping: {train_gen.class_indices}")
    
    num_classes = len(train_gen.class_indices)
    
    # Dictionary to store models and their histories
    models = {}
    histories = {}
    
    # Train baseline model if enabled
    if config["models"].get("baseline_model", True):
        print("\n===== Training Baseline Model =====")
        baseline_model = create_baseline_model(config, num_classes)
        baseline_history = train_model(config, baseline_model, train_gen, val_gen)
        models["Baseline"] = baseline_model
        histories["Baseline"] = baseline_history
        
        # Plot training history
        if config["output"].get("save_training_plots", True):
            plot_training_history(
                baseline_history,
                "Baseline",
                save_dir=config["output"]["results_dir"]
            )
    
    # Train transfer learning models if specified
    transfer_models = config["models"].get("transfer_learning_models", [])
    
    for model_name in transfer_models:
        try:
            print(f"\n===== Training {model_name} Model =====")
            transfer_model = create_transfer_learning_model(config, model_name, num_classes)
            transfer_history = train_model(
                config, 
                transfer_model, 
                train_gen, 
                val_gen, 
                fine_tune=True
            )
            models[model_name] = transfer_model
            histories[model_name] = transfer_history
            
            # Plot training history
            if config["output"].get("save_training_plots", True):
                plot_training_history(
                    transfer_history,
                    model_name,
                    save_dir=config["output"]["results_dir"]
                )
        except Exception as e:
            print(f"Error training {model_name}: {e}")
    
    # Create ensemble model if enabled
    if config["models"].get("create_ensemble", True) and len(models) > 1:
        print("\n===== Creating Ensemble Model =====")
        
        # Get model paths
        model_paths = {}
        for name in models:
            if name == "Baseline":
                model_paths[name] = os.path.join(config["models"]["save_dir"], f"Baseline_best.h5")
            else:
                model_paths[name] = os.path.join(config["models"]["save_dir"], f"{name}_ft_best.h5")
        
        try:
            ensemble_model = create_ensemble_model(config, model_paths, num_classes)
            ensemble_history = train_model(config, ensemble_model, train_gen, val_gen)
            models["Ensemble"] = ensemble_model
            histories["Ensemble"] = ensemble_history
            
            # Plot training history
            if config["output"].get("save_training_plots", True):
                plot_training_history(
                    ensemble_history,
                    "Ensemble",
                    save_dir=config["output"]["results_dir"]
                )
        except Exception as e:
            print(f"Error creating ensemble model: {e}")
    
    # Evaluate all models
    print("\n===== Evaluating Models =====")
    results_df = evaluate_models(config, models, test_gen)
    
    print("\nResults summary:")
    print(results_df[['Model', 'Test Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']])
    
    # Plot model comparison
    if config["output"].get("save_training_plots", True) and len(models) > 1:
        plot_model_comparison(
            results_df,
            save_path=os.path.join(config["output"]["results_dir"], "model_comparison.png")
        )
    
    # Print best model
    if len(models) > 1:
        best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Test Accuracy']
        print(f"\nBest model: {best_model} with test accuracy: {best_accuracy:.4f}")

    print("\n===== Pipeline Completed =====")


if __name__ == "__main__":
    main()