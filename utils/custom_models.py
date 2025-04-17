from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn


__all__ = ["create_model_instance", "AlexNet", "CNNHAR"]



def create_model_instance(model_name, dataset_name, width=1.0):
    """
    创建不同宽度的模型, 包括0.25, 0.5, 0.75和1.0
    """
    if model_name == 'alexnet':
        if dataset_name == 'cifar10':
            model = AlexNet(num_classes=10, width=width)
        elif dataset_name == 'cifar100':
            model = AlexNet(num_classes=100, width=width)
    elif model_name == 'cnn':
        if dataset_name == 'har':
            model = CNNHAR(width=width)
    elif model_name == 'vgg':
        if dataset_name == 'imagenet':
            model = VGG16(num_classes=100, width=width)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5, width: float = 1.0) -> None:
        super().__init__()
        self.width = width  # 增加self.width属性防止'AlexNet' object has no attribute 'width'
        def adjust_channels(channels: int) -> int:
            return max(1, int(channels * width))
        self.features = nn.Sequential(
            # 3 32 32
            nn.Conv2d(3, adjust_channels(64), kernel_size=3, stride=2, padding=1),
            # 64 32+2-2=32 32/2=16
            nn.ReLU(inplace=True),
            # 64 16 16
            nn.MaxPool2d(kernel_size=2),
            # 64 8 8
            nn.Conv2d(adjust_channels(64), adjust_channels(192), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 192 8 8
            nn.MaxPool2d(kernel_size=2),
            # 192 4 4
            nn.Conv2d(adjust_channels(192), adjust_channels(384), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 384 4 4
            nn.Conv2d(adjust_channels(384), adjust_channels(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 256 4 4
            nn.Conv2d(adjust_channels(256), adjust_channels(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # 256 2 2
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(adjust_channels(256) * 2 * 2, adjust_channels(1024)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(adjust_channels(1024), adjust_channels(512)),
            nn.ReLU(inplace=True),
            nn.Linear(adjust_channels(512), num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # return F.log_softmax(x, dim=1)
        return x

class CNNHAR(nn.Module):
    def __init__(self, num_classes: int = 6, width: float = 1.0) -> None:
        super().__init__()
        def adjust_channels(channels: int) -> int:
            return max(1, int(channels * width))

        self.features = nn.Sequential(
            nn.Conv1d(9, adjust_channels(64), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(adjust_channels(64), adjust_channels(128), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(adjust_channels(128), adjust_channels(256), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(adjust_channels(256) * 12, adjust_channels(128)),
            nn.ReLU(inplace=True),
            nn.Linear(adjust_channels(128), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, width: float = 1.0) -> None:
        super().__init__()
        def adjust_channels(channels: int) -> int:
            return max(1, int(channels * width))

        self.features = nn.Sequential(
            nn.Conv2d(3, adjust_channels(64), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(64), adjust_channels(64), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(adjust_channels(64), adjust_channels(128), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(128), adjust_channels(128), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(adjust_channels(128), adjust_channels(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(256), adjust_channels(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(256), adjust_channels(256), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(adjust_channels(256), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(512), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(512), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(adjust_channels(512), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(512), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adjust_channels(512), adjust_channels(512), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(adjust_channels(512) * 7 * 7, adjust_channels(4096)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(adjust_channels(4096), adjust_channels(4096)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(adjust_channels(4096), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    