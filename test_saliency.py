import torch
import numpy as np
from model import SwinWaferModel
from visualize_attention import generate_saliency_map

model = SwinWaferModel(num_classes=9)
dummy_tensor = torch.randn(1, 1, 64, 360, requires_grad=True)
saliency = generate_saliency_map(model, dummy_tensor)
print("Saliency generated shape:", saliency.shape)
