import torch
import torch.nn as nn
import timm


class SwinTinyBinary(nn.Module):
    """
    SwinTinyBinary model for oral cancer classification
    Uses Swin Transformer tiny architecture from the timm library
    """
    def __init__(self, config):
        """
        Initialize the model with configuration parameters
        
        Args:
            config: Configuration dictionary with model parameters
        """
        super(SwinTinyBinary, self).__init__()
        
        # Get model parameters from config
        self.architecture = config['model'].get('architecture', 'swin_tiny_patch4_window7_224')
        self.pretrained = config['model'].get('pretrained', True)
        self.num_classes = config['model'].get('num_classes', 4)
        
        # Create the model
        self.model = timm.create_model(
            self.architecture, 
            pretrained=self.pretrained, 
            num_classes=self.num_classes
        )
        
        # Print model summary
        print(f"Initialized SwinTinyBinary with architecture: {self.architecture}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Pretrained: {self.pretrained}")

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Model output of shape (batch_size, num_classes)
        """
        return self.model(x)


def load_model(config, device):
    """
    Load the model based on the configuration
    
    Args:
        config: Configuration dictionary
        device: Device to load the model on
        
    Returns:
        Initialized model
    """
    model = SwinTinyBinary(config).to(device)
    
    # Load checkpoint if specified and exists
    checkpoint_path = config['model'].get('checkpoint_path', None)
    if checkpoint_path and checkpoint_path != "":
        try:
            if device.type == 'cuda':
                state_dict = torch.load(checkpoint_path)
            else:
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            model.load_state_dict(state_dict)
            print(f"Loaded model from checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return model