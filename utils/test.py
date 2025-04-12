from model_utils import *

model = create_model_instance('alexnet', 'cifar10', 1.0)
if model is None:
    print("Model creation failed.")
else:
    print("Model created successfully.")
    print(f"Model summary: {model}")