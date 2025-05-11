import os
import torch
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class OralCancerDataset(Dataset):
    """
    Dataset class for oral cancer images
    """
    def __init__(self, root_dir, split, transform=None, image_size=224):
        """
        Initialize the dataset
        
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', or 'test')
            transform: Transforms to apply to images
            image_size: Image size for resizing
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_size = image_size
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform - same as in data_loader.py
                transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                image = transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor in case of error
            return torch.zeros((3, self.image_size, self.image_size)), label


def get_transforms(config, split):
    """
    Get transforms for a specific dataset split based on configuration
    
    Args:
        config: Configuration dictionary
        split: Dataset split ('train', 'val', or 'test')
        
    Returns:
        Transforms for the specified split
    """
    image_size = config['data'].get('image_size', 224)
    
    # Simple transforms as in original data_loader.py
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    

def get_dataloaders(config):
    """
    Get data loaders for training, validation, and testing
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing data loaders for train, val, and test splits
    """
    data_dir = config['data']['data_dir']
    batch_size = config['data']['batch_size']
    num_workers = config['data'].get('num_workers', 4)
    
    # Use the same transformations as in data_loader.py
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    loaders = {}
    
    # Training dataloader - use ImageFolder like in original data_loader.py
    train_dir = config['data'].get('train_dir', 'augmented_train')
    if os.path.exists(os.path.join(data_dir, train_dir)):
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, train_dir), transform=transform)
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Validation dataloader - use ImageFolder like in original data_loader.py
    val_dir = config['data'].get('val_dir', 'val')
    if os.path.exists(os.path.join(data_dir, val_dir)):
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, val_dir), transform=transform)
        loaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test dataloader - use ImageFolder like in original data_loader.py
    test_dir = config['data'].get('test_dir', 'test')
    if os.path.exists(os.path.join(data_dir, test_dir)):
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, test_dir), transform=transform)
        loaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return loaders