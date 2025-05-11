"""
Visualization utilities for the Oral Cancer Classification Pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Any


def plot_class_distribution(original_counts: Dict[str, int], augmented_counts: Dict[str, int] = None, save_path: str = None):
    """
    Plot the class distribution before and after augmentation.
    
    Args:
        original_counts: Dictionary of original class counts
        augmented_counts: Optional dictionary of augmented class counts
        save_path: Path to save the plot (if specified)
    """
    classes = list(original_counts.keys())
    original_values = list(original_counts.values())
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(classes))
    bar_width = 0.35
    
    plt.bar(x - bar_width/2 if augmented_counts else x, original_values, bar_width, label='Original')
    
    if augmented_counts:
        augmented_values = [augmented_counts[cls] for cls in classes]
        plt.bar(x + bar_width/2, augmented_values, bar_width, label='After Augmentation')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_augmentations(original_img, augmented_imgs, save_path: str = None):
    """
    Visualize original image alongside augmented versions.
    
    Args:
        original_img: Original image
        augmented_imgs: List of augmented images
        save_path: Path to save the plot (if specified)
    """
    n_augmented = len(augmented_imgs)
    n_total = n_augmented + 1  # +1 for original
    
    # Calculate grid dimensions
    n_cols = min(4, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(original_img)
    plt.title('Original')
    plt.axis('off')
    
    # Plot augmented images
    for i, img in enumerate(augmented_imgs):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(img)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = None):
    """
    Plot a confusion matrix.
    
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


def plot_training_history(history: Dict[str, List], model_name: str, save_dir: str = None):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{model_name}_history.png"))
        plt.close()
    else:
        plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Plot model comparison.
    
    Args:
        results_df: DataFrame containing model results
        save_path: Path to save the plot (if specified)
    """
    plt.figure(figsize=(14, 8))
    
    metrics = ['Test Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-Score (macro)']
    
    # Create a bar plot for each metric
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        ax = sns.barplot(x='Model', y=metric, data=results_df)
        plt.title(metric)
        plt.xticks(rotation=45)
        
        # Add value labels to bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.3f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()