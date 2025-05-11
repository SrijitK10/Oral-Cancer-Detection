"""
Evaluation utilities for the Oral Cancer Classification Pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple


def evaluate_model(model, test_gen, class_names: List[str], results_dir: str, model_name: str) -> Dict[str, float]:
    """
    Evaluate a model and generate performance metrics.
    
    Args:
        model: The model to evaluate
        test_gen: Test data generator
        class_names: List of class names
        results_dir: Directory to save results
        model_name: Name of the model
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Reset generator to ensure consistent evaluation
    test_gen.reset()
    
    # Evaluate model
    print(f"Evaluating {model_name}...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    
    # Generate predictions for confusion matrix and classification report
    test_gen.reset()
    predictions = model.predict(test_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a DataFrame for the confusion matrix
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Save confusion matrix and classification report
    results_path = os.path.join(results_dir, f"{model_name}_results.txt")
    
    with open(results_path, 'w') as f:
        f.write(f"Model: {model_name}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm_df) + "\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create a dictionary with evaluation metrics
    metrics = {
        'Model': model_name,
        'Test Loss': test_loss,
        'Test Accuracy': test_acc,
        'Precision (macro)': report['macro avg']['precision'],
        'Recall (macro)': report['macro avg']['recall'],
        'F1-Score (macro)': report['macro avg']['f1-score']
    }
    
    # Add per-class metrics
    for class_name in class_names:
        metrics[f'{class_name} Precision'] = report[class_name]['precision']
        metrics[f'{class_name} Recall'] = report[class_name]['recall']
        metrics[f'{class_name} F1-Score'] = report[class_name]['f1-score']
    
    return metrics


def evaluate_models(config: Dict[str, Any], models: Dict[str, Any], test_gen) -> pd.DataFrame:
    """
    Evaluate multiple models on the test set.
    
    Args:
        config: Configuration dictionary
        models: Dictionary of model name to model
        test_gen: Test data generator
        
    Returns:
        DataFrame with evaluation results
    """
    print("\nEvaluating models...")
    results = []
    
    # Get class names
    class_names = list(test_gen.class_indices.keys())
    results_dir = config["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    for name, model in models.items():
        metrics = evaluate_model(model, test_gen, class_names, results_dir, name)
        results.append(metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(results_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    
    return results_df


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot (if specified)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()