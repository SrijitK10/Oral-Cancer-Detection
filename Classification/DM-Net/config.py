import torch

class Config:
    data_dir = ".\organized_dataset"  # Update this
    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    checkpoint_path = "best_swin_tiny.pth"
    num_classes = 4  # Binary classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
