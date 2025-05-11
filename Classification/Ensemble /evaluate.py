"""
Model evaluation script for the Oral Cancer Classification Pipeline.

This script focuses on evaluating pre-trained models:
- Loading models from saved files
- Detailed evaluation on test data
- Performance visualization and comparison
- Support for single model or ensemble evaluation

Example usage:
    python evaluate.py --model ./models/Baseline_best.h5
    python evaluate.py --model ./models/MobileNetV2_ft_best.h5 --data_dir ./test_data
    python evaluate.py --ensemble --models_dir ./models
"""

import os
import sys
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import project modules
from src.utils.config_utils import load_config, parse_arguments, merge_configs
from src.data.data_loader import create_data_generators
from src.models.evaluation import evaluate_models, plot_confusion_matrix
from src.visualization.visualization import plot_model_comparison


def parse_eval_arguments():
    """
    Parse command line arguments for evaluation.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Oral Cancer Classification Model Evaluation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="./config/config.yml", 
        help="Path to the configuration YAML file"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Path to the model file to evaluate"
    )
    
    parser.add_argument(
        "--ensemble", 
        action="store_true", 
        help="Evaluate all models and create ensemble comparison"
    )
    
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="./models", 
        help="Directory containing model files for ensemble evaluation"
    )
    
    parser.add_argument(
        "--data_dir", 
        type=str, 
        help="Override dataset directory from config"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--detailed", 
        action="store_true", 
        help="Generate detailed performance analysis"
    )
    
    parser.add_argument(
        "--save_plots", 
        action="store_true", 
        help="Save visualization plots"
    )
    
    return parser.parse_args()


def evaluate_single_model(model_path, test_gen, config, results_dir, detailed=False, save_plots=False):
    """
    Evaluate a single model.
    
    Args:
        model_path: Path to the model file
        test_gen: Test data generator
        config: Configuration dictionary
        results_dir: Directory to save results
        detailed: Whether to generate detailed analysis
        save_plots: Whether to save plots
    """
    # Extract model name from path
    model_name = os.path.basename(model_path).split('_')[0]
    print(f"\nEvaluating model: {model_name}")
    
    # Load model
    model = load_model(model_path)
    
    # Evaluate model
    print("Running evaluation...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    # Generate classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Plot confusion matrix
    if save_plots:
        cm_plot_path = os.path.join(results_dir, f"{model_name}_cm.png")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"Confusion matrix saved to: {cm_plot_path}")
    
    # Save results to file
    results_path = os.path.join(results_dir, f"{model_name}_evaluation.txt")
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_path}\n\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm_df))
    
    print(f"Evaluation results saved to: {results_path}")
    
    if detailed:
        # Generate per-class metrics
        print("\nPer-class analysis:")
        
        for i, class_name in enumerate(class_names):
            # Calculate true positives, false positives, etc.
            true_pos = cm[i, i]
            false_pos = cm[:, i].sum() - true_pos
            false_neg = cm[i, :].sum() - true_pos
            true_neg = cm.sum() - (true_pos + false_pos + false_neg)
            
            # Calculate metrics
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0
            
            print(f"\n{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1_score:.4f}")
            print(f"  Specificity: {specificity:.4f}")


def evaluate_ensemble(models_dir, test_gen, config, results_dir, save_plots=False):
    """
    Evaluate multiple models and compare performance.
    
    Args:
        models_dir: Directory containing model files
        test_gen: Test data generator
        config: Configuration dictionary
        results_dir: Directory to save results
        save_plots: Whether to save plots
    """
    print(f"\nEvaluating models in directory: {models_dir}")
    
    # Find all model files
    model_files = [
        os.path.join(models_dir, f) 
        for f in os.listdir(models_dir) 
        if f.endswith('.h5')
    ]
    
    if not model_files:
        print(f"No model files found in {models_dir}")
        sys.exit(1)
    
    # Load models
    models = {}
    for model_path in model_files:
        try:
            model_name = os.path.basename(model_path).split('_')[0]
            models[model_name] = load_model(model_path)
            print(f"Loaded model: {model_name} from {model_path}")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    if not models:
        print("No models could be loaded.")
        sys.exit(1)
    
    # Evaluate models
    print("Evaluating models...")
    results_df = evaluate_models(config, models, test_gen)
    
    print("\nResults summary:")
    print(results_df[['Model', 'Test Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']])
    
    # Plot model comparison
    if save_plots:
        comparison_plot_path = os.path.join(results_dir, "model_comparison.png")
        plot_model_comparison(results_df, save_path=comparison_plot_path)
        print(f"Model comparison plot saved to: {comparison_plot_path}")
    
    # Print best model
    best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
    best_accuracy = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Test Accuracy']
    print(f"\nBest model: {best_model} with test accuracy: {best_accuracy:.4f}")


def main():
    """Main entry point for the evaluation pipeline."""
    args = parse_eval_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config values with command line args
    if args.data_dir:
        config["dataset"]["base_dir"] = args.data_dir
    if args.batch_size:
        config["dataset"]["batch_size"] = args.batch_size
    if args.output_dir:
        config["output"]["results_dir"] = args.output_dir
    
    # Set up results directory
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    print("\n===== Oral Cancer Classification Model Evaluation =====")
    print(f"Configuration: {args.config}")
    print(f"Dataset: {config['dataset']['base_dir']}")
    
    # Create data generators (test only)
    _, _, test_gen = create_data_generators(config, use_augmented=False)
    
    print(f"Test set: {test_gen.samples} images")
    print(f"Number of classes: {len(test_gen.class_indices)}")
    print(f"Class mapping: {test_gen.class_indices}")
    
    # Evaluate single model or ensemble
    if args.model:
        # Single model evaluation
        evaluate_single_model(
            args.model,
            test_gen,
            config,
            results_dir,
            detailed=args.detailed,
            save_plots=args.save_plots
        )
    elif args.ensemble:
        # Evaluate all models in directory
        evaluate_ensemble(
            args.models_dir,
            test_gen,
            config,
            results_dir,
            save_plots=args.save_plots
        )
    else:
        print("Please specify either --model or --ensemble")
        sys.exit(1)
    
    print("\n===== Evaluation Completed =====")


if __name__ == "__main__":
    main()