import torch
import streamlit as st
from model import MultiHeadResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = "clean_model.pth"

@st.cache_resource
def load_model():
    st.write("🔄 Loading model...")
    model = MultiHeadResNet(backbone_name="resnet50", pretrained=False).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # Handle model checkpoints stored as dict
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    # Filter out mismatched keys safely
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model.state_dict() and model.state_dict()[k].shape == v.shape:
            new_state_dict[k] = v
        else:
            print(f"Skipping {k} due to mismatch: {v.shape if hasattr(v, 'shape') else 'N/A'}")

    # Load safely
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    st.write(f"✅ Model loaded successfully with {len(new_state_dict)} matching layers.")
    if missing:
        st.warning(f"Missing keys: {missing}")
    if unexpected:
        st.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    return model


# Test load
model = load_model()
st.success("🎉 Model initialized and ready for inference!")
