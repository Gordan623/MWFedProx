import torch
import sys
import time
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
import torchvision
import torch.nn.functional as F
import random
import os
import copy

class LeNet_Emnist(nn.Module):
    def __init__(self, width):
        super(LeNet_Emnist, self).__init__()
        self.width = width
        self.features = nn.Sequential(
            # input size: 1*28*28
            nn.Conv2d(1, int(6*width), kernel_size=5, stride=1, padding=2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # output size: 6*14*14
            
            # input size: 6*14*14
            nn.Conv2d(int(6*width), int(16*width), kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # output size: 16*5*5
        )
        self.classifier = nn.Sequential(
            # input size: 16*5*5
            nn.Linear(int(16*width) * 5 * 5, int(120*width)),
            nn.ReLU(),

            # input size: 120
            nn.Linear(int(120*width), int(84*width)),
            nn.ReLU(),
            
            # input size: 84
            nn.Linear(int(84*width), class_num = 62)
        )
 
    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.features(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        return x

