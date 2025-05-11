# DM-Net: Oral Cancer Classification Pipeline

A comprehensive pipeline for training, evaluating, and using deep learning models for oral cancer classification.

## Project Structure

```
DM-Net/
├── config/                    # Configuration files
│   ├── default.yml            # Default configuration
│   ├── train_config.yml       # Training-specific configuration
│   └── test_config.yml        # Testing-specific configuration
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── models/                # Model architectures
│   ├── training/              # Training functionality
│   ├── evaluation/            # Evaluation functionality
│   └── utils/                 # Utility functions
├── main.py                    # Main entry point
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Features

- **Modular Design**: Clear separation of concerns for better maintainability
- **YAML Configuration**: Flexible configuration via YAML files
- **Command Line Interface**: Override configurations via CLI arguments
- **Comprehensive Evaluation**: Multiple metrics for model performance assessment
- **Visualization**: Training curves, confusion matrices, ROC curves
- **Reproducibility**: Seed setting for consistent results

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd DM-Net
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

The pipeline uses a hierarchical configuration system:

1. **Default config** (`config/default.yml`) - Base configuration
2. **Mode-specific config** (`config/train_config.yml` or `config/test_config.yml`)
3. **CLI arguments** - Override specific parameters from the command line

### Configuration Options

Key configuration options include:

#### Data Configuration
- `data_dir`: Root directory for datasets
- `batch_size`: Batch size for data loading
- `image_size`: Size to which images are resized

#### Model Configuration
- `architecture`: Model architecture to use (via timm library)
- `num_classes`: Number of output classes
- `pretrained`: Whether to use pretrained weights

#### Training Configuration
- `num_epochs`: Number of training epochs
- `learning_rate`: Initial learning rate
- `optimizer`: Optimizer type (adam, adamw, sgd)
- `scheduler`: Learning rate scheduler

## Usage

### Training

Train a model using default configuration:
```bash
python main.py --mode train
```

Train with a custom configuration:
```bash
python main.py --mode train --data-dir ./my_dataset --batch-size 64 --lr 0.0001
```

### Evaluation

Evaluate a trained model:
```bash
python main.py --mode test --checkpoint-path ./checkpoints/best_model.pth
```

### Prediction

Generate predictions for a dataset:
```bash
python main.py --mode predict --checkpoint-path ./checkpoints/best_model.pth --data-dir ./test_data
```

## Examples

### Train a model with custom settings

```bash
python main.py --mode train \
    --data-dir ./my_dataset \
    --batch-size 32 \
    --epochs 50 \
    --lr 0.0001 \
    --model swin_tiny_patch4_window7_224 \
    --optimizer adamw
```

### Test a trained model

```bash
python main.py --mode test \
    --checkpoint-path ./checkpoints/best_model.pth \
    --data-dir ./test_data \
    --batch-size 64
```

## License

[Specify your license]