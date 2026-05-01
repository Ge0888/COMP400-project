import torch
import torchvision.models as tvmodels

model = tvmodels.regnet_y_128gf()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

print("Total params:", total_params)
print("Trainable params:", trainable_params)
print("Total parameter bytes:", total_param_bytes)

# Command: python regnet_model_size.py