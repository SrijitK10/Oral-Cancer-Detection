import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Trainer:
    """
    Trainer class for SwinTinyBinary oral cancer classification model
    """
    def __init__(self, model, dataloaders, config):
        """
        Initialize the trainer
        
        Args:
            model: The model to train
            dataloaders: Dictionary containing data loaders for train and val splits
            config: Configuration dictionary
        """
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        
        # Extract device from config
        self.device = torch.device(config['runtime'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set up training parameters
        self.num_epochs = config['training'].get('num_epochs', 30)
        self.learning_rate = config['training'].get('learning_rate', 1e-4)
        
        # Set up loss function - using CrossEntropyLoss as in the original
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up optimizer - using AdamW as in the original without explicit weight decay
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        # Create checkpoint directory if needed
        self.checkpoint_dir = config['training'].get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, config['model'].get('checkpoint_path', 'best_model.pth'))
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        """
        Train the model for one epoch - following the original train.py logic
        
        Returns:
            Training loss and accuracy for this epoch
        """
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        
        progress_bar = tqdm(self.dataloaders['train'], desc=f"Training")
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Collect statistics - using same calculation as the original
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
        
        # Calculate epoch metrics - using same approach as the original
        epoch_loss = train_loss / len(self.dataloaders['train'])
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model on the validation set - following the original train.py logic
        
        Returns:
            Validation loss and accuracy
        """
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in self.dataloaders['val']:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Collect statistics
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / len(self.dataloaders['val'])
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self):
        """
        Training loop for the model - matching the original train.py flow
        
        Returns:
            Training history (losses and accuracies)
        """
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Print epoch results - same format as original
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model - same logic as original
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.best_model_path)
                print("Checkpoint Saved!")
        
        # Plot training history
        self.plot_training_history()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
    
    def plot_training_history(self):
        """
        Plot training and validation losses and accuracies
        """
        plot_dir = self.config['visualization'].get('plot_dir', 'training_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot metrics using the same approach as the original utils.py
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'training_history.png'))
        plt.close()
        
        print(f"Training history plots saved to {plot_dir}/training_history.png")