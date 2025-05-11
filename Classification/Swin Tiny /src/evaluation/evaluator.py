import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)


class Evaluator:
    """
    Evaluator class for DM-Net oral cancer classification model
    """
    def __init__(self, model, dataloader, config):
        """
        Initialize the evaluator
        
        Args:
            model: Trained model to evaluate
            dataloader: DataLoader for the test dataset
            config: Configuration dictionary
        """
        self.model = model
        self.dataloader = dataloader
        self.config = config
        
        # Extract evaluation parameters
        self.device = torch.device(config['runtime'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.metrics = config['evaluation'].get('metrics', ['accuracy', 'precision', 'recall', 'f1'])
        self.output_dir = config['evaluation'].get('output_dir', 'results')
        self.save_predictions = config['evaluation'].get('save_predictions', False)
        self.predictions_file = config['evaluation'].get('predictions_file', 'predictions.csv')
        
        # Make sure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate(self):
        """
        Evaluate the model on the test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_file_paths = []  # For storing file paths if available
        
        with torch.no_grad():
            with tqdm(self.dataloader, desc="Evaluating") as progress_bar:
                for batch in progress_bar:
                    # Handle if batch contains file paths
                    if len(batch) == 3:
                        images, labels, file_paths = batch
                        all_file_paths.extend(file_paths)
                    else:
                        images, labels = batch
                    
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    
                    # Collect predictions and labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        results = {}
        
        # Basic metrics
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(all_labels, all_preds)
        if 'precision' in self.metrics:
            results['precision'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        if 'recall' in self.metrics:
            results['recall'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        if 'f1' in self.metrics:
            results['f1'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Confusion matrix
        if 'confusion_matrix' in self.metrics:
            cm = confusion_matrix(all_labels, all_preds)
            results['confusion_matrix'] = cm
            self.plot_confusion_matrix(cm)
        
        # ROC curve (for binary classification or one-vs-rest for multiclass)
        if 'roc_curve' in self.metrics:
            self.plot_roc_curve(all_labels, all_probs)
        
        # Classification report
        results['classification_report'] = classification_report(
            all_labels, all_preds, zero_division=0, output_dict=True
        )
        
        # Display results
        print("\nTest Results:")
        for metric, value in results.items():
            if metric not in ['confusion_matrix', 'classification_report']:
                print(f"{metric.capitalize()}: {value:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        
        # Save predictions if requested
        if self.save_predictions and len(all_file_paths) > 0:
            self.save_prediction_results(all_file_paths, all_labels, all_preds, all_probs)
        
        return results
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
        """
        plot_dir = self.config['visualization'].get('plot_dir', 'test_visualizations')
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Set class labels
        num_classes = cm.shape[0]
        classes = [str(i) for i in range(num_classes)]
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
        plt.close()
        print(f"Confusion matrix saved to {plot_dir}/confusion_matrix.png")
    
    def plot_roc_curve(self, y_true, y_score):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_score: Predicted probabilities
        """
        plot_dir = self.config['visualization'].get('plot_dir', 'test_visualizations')
        os.makedirs(plot_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        
        # Get number of classes from the shape of y_score
        n_classes = y_score.shape[1]
        
        # If binary classification, just plot one curve
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        else:
            # For multiclass, plot one-vs-rest ROC curves
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(plot_dir, 'roc_curve.png'))
        plt.close()
        print(f"ROC curve saved to {plot_dir}/roc_curve.png")
    
    def save_prediction_results(self, file_paths, true_labels, pred_labels, probabilities):
        """
        Save prediction results to a CSV file
        
        Args:
            file_paths: Paths to the image files
            true_labels: True labels
            pred_labels: Predicted labels
            probabilities: Predicted class probabilities
        """
        results = {
            'file_path': file_paths,
            'true_label': true_labels,
            'predicted_label': pred_labels,
        }
        
        # Add probability columns for each class
        for i in range(probabilities.shape[1]):
            results[f'prob_class_{i}'] = probabilities[:, i]
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(results)
        output_path = os.path.join(self.output_dir, self.predictions_file)
        df.to_csv(output_path, index=False)
        print(f"Prediction results saved to {output_path}")