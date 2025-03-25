import torch
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
sys.path.append(".")
sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append('../FLamingo/')
from models import AlexNet


def create_model_instance(model_type, dataset_type, width = 1.0):
    """
    Create model instance by passing model type and dataset type.
    The model and dataset must collaborate, else will return None.
    Use init_model to initialize your own model structure.
    Or submitting pull request to add your own.
    """
    # print("My self-defined function")
    assert model_type and dataset_type is not None, f"Model type and dataset type cannot be None."
    models_dict = {      
        ('mnist', 'alexnet'): AlexNet.AlexNetMnist,
        ('cifar10', 'alexnet'): AlexNet.AlexNet,
        ('cifar10', 'alexnet2'): AlexNet.AlexNet2,    
        ('image100', 'alexnet'): AlexNet.AlexNet_IMAGE,       
    }

    # 检查模型是否支持多宽度
    model = models_dict.get((dataset_type, model_type), None)
    if model:
        if hasattr(model, '__init__') and 'width' in model.__init__.__code__.co_varnames:
            return model(width = width)
        else:
            return model()
    else:
        return None
