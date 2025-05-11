# Oral Cancer Detection with YOLOv11

This project uses YOLOv11 to detect regions of interest in oral cavity images and classify them into four categories:
- Healthy
- Benign
- Oral Potentially Malignant Disorders (OPMD)
- Oral Cancer (OCA)

## Dataset

The dataset consists of 3,000 high-quality images of oral cavities taken with mobile phone cameras from the Sri Lankan population. The dataset includes:

- Images of oral cavities
- Annotations for oral cavity and lesion boundaries in COCO format
- Patient metadata including age, sex, diagnosis, and risk factors like smoking, alcohol consumption, and betel quid chewing

## Pipeline Components

1. **Data Preprocessing**: Convert COCO annotations to YOLO format
2. **Dataset Splitting**: Split data into training, validation and test sets
3. **YOLOv12 Training**: Train YOLOv12n model for oral cavity and lesion detection
4. **Model Evaluation**: Evaluate model performance on test data
5. **Inference**: Deploy model for real-time inference

## Setup

```bash
# Clone the repository
git clone <repository_url>
cd oral_cancer_detection

# Install dependencies
pip install -r requirements.txt

# Prepare the dataset
python utils/prepare_dataset.py --data_path /path/to/dataset

# Train the model
python train.py --config configs/yolov11.yaml

# Run inference
python predict.py --model models/best.pt --img /path/to/image.jpg
```

## Project Structure

```
oral_cancer_detection/
├── configs/           # Configuration files
├── data/              # Dataset files
├── models/            # Trained models
├── results/           # Training and evaluation results
├── utils/             # Utility scripts
│   ├── prepare_dataset.py
│   ├── coco_to_yolo.py
│   └── data_analysis.py
├── train.py           # Training script
├── evaluate.py        # Evaluation script
├── predict.py         # Inference script
└── requirements.txt   # Dependencies
```

## Results

[To be filled with model performance metrics and visualizations]
