import os
import torch
import numpy as np
import random
from src.utils.config_parser import parse_cli_args, get_config_from_args
from src.data.dataset import get_dataloaders
from src.models.model import load_model
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def set_seed(seed):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    """
    Main entry point for the DM-Net Oral Cancer Classification pipeline
    """
    # Parse command line arguments
    args = parse_cli_args()
    
    # Get configuration
    config = get_config_from_args(args)
    
    # Set random seed
    set_seed(config['runtime']['seed'])
    
    # Determine device
    device = torch.device(config['runtime']['device'])
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading datasets...")
    dataloaders = get_dataloaders(config)
    
    # Load model
    print("Loading model...")
    model = load_model(config, device)
    
    # Run in the specified mode
    if config['runtime']['mode'] == 'train':
        run_training(model, dataloaders, config)
    elif config['runtime']['mode'] == 'test':
        run_evaluation(model, dataloaders, config)
    elif config['runtime']['mode'] == 'predict':
        run_prediction(model, dataloaders, config)
    else:
        print(f"Unknown mode: {config['runtime']['mode']}")


def run_training(model, dataloaders, config):
    """
    Run training mode
    
    Args:
        model: Model to train
        dataloaders: Dictionary containing data loaders
        config: Configuration dictionary
    """
    print("\n=== Running in training mode ===")
    
    if 'train' not in dataloaders or 'val' not in dataloaders:
        print("Error: Training requires both train and validation datasets")
        return
    
    # Create and run trainer
    trainer = Trainer(model, dataloaders, config)
    history = trainer.train()
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f}")
    
    # Evaluate on test set if available
    if 'test' in dataloaders:
        print("\nEvaluating on test set...")
        evaluator = Evaluator(model, dataloaders['test'], config)
        evaluator.evaluate()


def run_evaluation(model, dataloaders, config):
    """
    Run evaluation mode
    
    Args:
        model: Model to evaluate
        dataloaders: Dictionary containing data loaders
        config: Configuration dictionary
    """
    print("\n=== Running in evaluation mode ===")
    
    if 'test' not in dataloaders:
        print("Error: Evaluation requires test dataset")
        return
    
    # Create and run evaluator
    evaluator = Evaluator(model, dataloaders['test'], config)
    results = evaluator.evaluate()
    
    # Print summary of results
    print("\nEvaluation completed!")
    if 'accuracy' in results:
        print(f"Test accuracy: {results['accuracy']:.4f}")


def run_prediction(model, dataloaders, config):
    """
    Run prediction mode (similar to evaluation but focused on generating predictions)
    
    Args:
        model: Model to use for prediction
        dataloaders: Dictionary containing data loaders
        config: Configuration dictionary
    """
    print("\n=== Running in prediction mode ===")
    
    # For now, just run evaluation with prediction saving enabled
    config['evaluation']['save_predictions'] = True
    
    if 'test' in dataloaders:
        evaluator = Evaluator(model, dataloaders['test'], config)
        evaluator.evaluate()
    else:
        print("Error: Prediction requires test dataset")


if __name__ == "__main__":
    main()