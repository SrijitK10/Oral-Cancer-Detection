import torch.nn as nn
import timm
from config import Config

class SwinTinyBinary(nn.Module):
    def __init__(self):
        super(SwinTinyBinary, self).__init__()
        self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=Config.num_classes)

    def forward(self, x):
        return self.model(x)