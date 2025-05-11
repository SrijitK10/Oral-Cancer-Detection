# Oral Cancer Detection System

A comprehensive system for oral cancer detection, combining region of interest (ROI) extraction and multi-class classification.

## Project Overview

This project implements an end-to-end pipeline for oral cancer detection from oral cavity images. It consists of two primary modules:

1. **ROI Extraction**: Uses YOLOv11/YOLOv12 to detect and segment regions of interest in oral cavity images
2. **Classification**: Employs multiple deep learning models, including an ensemble approach and the specialized DM-Net architecture, to classify detected regions into four categories:v
   - Healthy
   - Benign
   - Oral Potentially Malignant Disorders (OPMD)
   - Oral Cancer (OCA)

## Dataset

The dataset consists of 3,000 high-quality images of oral cavities captured with mobile phone cameras from the Sri Lankan population, including:

- Oral cavity images
- Annotations for oral cavity and lesion boundaries
- Patient metadata (age, sex, diagnosis, risk factors including smoking, alcohol consumption, and betel quid chewing)

## Project Structure

```
Oral Cancer Detection/
├── ROI Extraction/             # ROI extraction module
│   ├── configs/                # Configuration files for YOLO models
│   ├── data/                   # Dataset and processed data
│   ├── results/                # Analysis plots and evaluation results
│   ├── utils/                  # Utility functions for data processing
│   ├── train.py                # Training script for YOLO models
│   ├── predict.py              # Inference script
│   ├── evaluate.py             # Evaluation script
│   └── run_pipeline.py         # End-to-end pipeline script
├── Classification/             # Classification module
│   ├── Ensemble/               # Ensemble classification approach
│   │   ├── config/             # Configuration files
│   │   ├── results/            # Results and visualizations
│   │   ├── src/                # Source code
│   │   ├── train.py            # Training script
│   │   └── evaluate.py         # Evaluation script
│   └── DM-Net/                 # DM-Net classification architecture
│       ├── config/             # Configuration files
│       ├── src/                # Source code
│       ├── training_plots/     # Training visualizations
│       └── main.py             # Main entry point
└── README.md                   # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd oral-cancer-detection
   ```

2. Install dependencies for ROI Extraction:
   ```bash
   cd "ROI Extraction"
   pip install -r requirements.txt
   ```

3. Install dependencies for Classification modules:
   ```bash
   cd ../Classification/DM-Net
   pip install -r requirements.txt
   ```

## Usage

### 1. ROI Extraction Module

#### Data Preparation
```bash
cd "ROI Extraction"
python utils/prepare_dataset.py --data_path /path/to/dataset
```

#### Training
```bash
python train.py --config configs/yolov12.yaml
```

Additional parameters:
- `--data`: Override dataset path
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--img_size`: Input image size
- `--device`: Training device (cuda/cpu)
- `--weights`: Pre-trained weights

#### Evaluation
```bash
python evaluate.py --model models/best.pt --data data/processed/dataset.yaml
```

#### Inference
```bash
python predict.py --model models/best.pt --img /path/to/image.jpg
```

#### Full Pipeline
```bash
python run_pipeline.py --roi_model models/best.pt --classify_model ../Classification/DM-Net/checkpoints/best_model.pth
```

### 2. Classification Module

#### DM-Net

Training:
```bash
cd ../Classification/DM-Net
python main.py --mode train
```

Evaluation:
```bash
python main.py --mode test --checkpoint-path ./checkpoints/best_model.pth
```

Prediction:
```bash
python main.py --mode predict --checkpoint-path ./checkpoints/best_model.pth --data-dir ./test_data
```

#### Ensemble Approach

Training:
```bash
cd ../Classification/Ensemble
python train.py --config ./config/config.yml
```

Evaluation:
```bash
python evaluate.py --model ./models/model_name.h5
```

Ensemble evaluation:
```bash
python evaluate.py --ensemble --models_dir ./models
```

## Configuration

The project uses a hierarchical configuration system with YAML files:

### ROI Extraction
Configuration files in `ROI Extraction/configs/` control:
- Model architecture and weights
- Dataset paths
- Training hyperparameters
- Augmentation settings

### Classification
Configuration files in `Classification/DM-Net/config/` and `Classification/Ensemble/config/` control:
- Model architectures
- Learning rates and optimizers
- Batch size and image dimensions
- Augmentation parameters
- Evaluation metrics

## Results and Visualizations

### ROI Extraction
The ROI extraction module generates various analysis plots in the `ROI Extraction/results/` directory:
- Age and gender distribution
- Diagnosis distribution
- Risk factor analysis
- Annotation statistics

### Classification
Classification results are stored in:
- `Classification/DM-Net/training_plots/`: Confusion matrices and ROC curves
- `Classification/Ensemble/results/`: Model comparison metrics and history plots

## Performance

The system achieves the following performance metrics:
- ROI extraction: [Add metrics when available]
- Classification: [Add metrics when available]
- End-to-end pipeline: [Add metrics when available]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Specify the license here]

## Acknowledgements

- [Add acknowledgements]