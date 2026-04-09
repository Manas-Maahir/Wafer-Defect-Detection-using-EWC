import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import HybridWaferModel
from preprocessing import preprocess_wafer

def generate_saliency_map(model, input_tensor):
    """
    Generates a simple saliency map (input gradients w.r.t. the max-score class).
    """
    model.eval()
    input_tensor.requires_grad_()
    
    output = model(input_tensor)
    score, _ = torch.max(output, 1)
    score.backward()
    
    # Saliency is the absolute value of gradients
    saliency, _ = torch.max(torch.abs(input_tensor.grad.data), dim=1)
    saliency = saliency.reshape(input_tensor.shape[2], input_tensor.shape[3])
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.cpu().numpy()

def visualize_defect_attention(wafer_map, model_path=None):
    """
    Full visualization: Input -> Polar -> Saliency Map
    """
    # 1. Preprocess
    polar_strip = preprocess_wafer(wafer_map)
    input_tensor = torch.from_numpy(polar_strip).float().unsqueeze(0).unsqueeze(0)
    
    # 2. Load Model (or use dummy for demo)
    model = HybridWaferModel(num_classes=9)
    if model_path:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        # Support both raw state-dicts and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    
    # 3. Generate Map
    saliency = generate_saliency_map(model, input_tensor)
    
    # 4. Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(polar_strip, cmap='viridis')
    axes[0].set_title("Polar Transformed Edge Ring")
    
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title("Defect Attention (Saliency Map)")
    
    plt.tight_layout()
    plt.savefig('wafer_attention.png')
    print("Attention map saved to wafer_attention.png")

if __name__ == "__main__":
    # Test with dummy data
    dummy_wafer = np.zeros((100, 100))
    cv2.circle(dummy_wafer, (50, 50), 45, 2, 2) # Synthetic edge ring defect
    visualize_defect_attention(dummy_wafer)
