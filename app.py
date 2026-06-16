import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from preprocessing import preprocess_wafer
from model import SwinWaferModel

# --- Configuration ---
CHECKPOINT_PATH = "checkpoints/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full", "none"
]

# --- Saliency Function ---
def get_saliency_map(model, input_tensor):
    model.eval()
    input_tensor.requires_grad_()
    
    logits = model(input_tensor)
    score, pred_class = torch.max(logits, 1)
    
    model.zero_grad()
    score.backward()
    
    saliency, _ = torch.max(torch.abs(input_tensor.grad.data), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency

# --- Load Model ---
@st.cache_resource
def load_wafer_model():
    model = SwinWaferModel(num_classes=len(CLASS_NAMES)).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model

# --- Preprocessing Interface ---
def image_to_wafer_map(image_array):
    """
    Given a grayscale numpy image, convert it to a 0, 1, 2 mask.
    Background=0, Normal Die=1, Defect=2.
    """
    out = np.zeros_like(image_array, dtype=np.uint8)
    out[image_array > 10] = 1   # Normal dies
    out[image_array > 200] = 2  # Defects (bright pixels)
    
    return out

# --- UI Setup ---
st.set_page_config(page_title="Wafer Defect Analyzer", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1 {color: #00d2ff;}
    .stAlert {background-color: #1e2130;}
    .css-1aumxhk {background-color: #1e2130;}
    </style>
""", unsafe_allow_html=True)

st.title(" Wafer Defect Analysis Dashboard")
st.markdown("Upload a Wafer Image to detect and classify defects using the internally trained Vision Transformer.")

# Wait until model is loaded
with st.spinner("Loading Wafer Defect Model..."):
    model = load_wafer_model()

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.success(f"Model Engine Active (`{DEVICE.upper()}`)")
st.sidebar.info("This system analyzes edge-ring defects by polar-transforming the wafer edge.")

upload = st.file_uploader("Upload Wafer Image (PNG, JPG)", type=['png', 'jpg', 'jpeg'])

if upload is not None:
    # 1. Read Image
    pil_img = Image.open(upload)
    img_array = np.array(pil_img)

    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Input Image")
        st.image(img_array, use_container_width=True, caption="Uploaded Wafer")

    # 2. Convert to wafer map
    wafer_map = image_to_wafer_map(gray)
        
    # 3. Extract Polar Strip
    polar_strip = preprocess_wafer(wafer_map)
    # Normalize pixel values so 0,1,2 becomes 0.0, 0.5, 1.0
    polar_strip = polar_strip.astype(np.float32) / 2.0
        
    with col2:
        st.subheader("2. Polar Edge Ring")
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.imshow(polar_strip, cmap='viridis')
        ax.axis("off")
        st.pyplot(fig)

    st.divider()

    # 4. Inference
    with st.spinner("Running Swin Transformer Analysis..."):
        input_tensor = torch.from_numpy(polar_strip).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx] * 100

    st.subheader("3. Analysis Results")
    if pred_class == "none":
        st.success(f"✅ **PASSED: Wafer is Defect Free!**")
    else:
        st.error(f"⚠️ **FAILED: Defect Detected ➔ {pred_class}**")
        
        # 5. Saliency Map
        st.markdown(f"#### Defect Localization")
        with st.spinner("Generating attention map..."):
            saliency = get_saliency_map(model, input_tensor)
            
        fig2, ax2 = plt.subplots(figsize=(10, 3))
        ax2.imshow(polar_strip, cmap='gray')
        # Overlay hot colormap where saliency is high
        # We can use a masked array or just alpha
        ax2.imshow(saliency, cmap='hot', alpha=0.5)
        ax2.axis("off")
        st.pyplot(fig2)

        st.caption("Attention Map: Hotter colors indicate the regions that influenced the model's defect prediction.")
