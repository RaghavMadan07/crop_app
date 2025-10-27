import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import MultiHeadResNet

# Load model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "best_multitask_model.pth"

st.title("🌾 Crop Condition Prediction App")

@st.cache_resource
@st.cache_resource
def load_model():
    model = MultiHeadResNet(pretrained=False, num_crops=5, num_stages=3, num_severity=4)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # Handle both formats safely
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        le_crop = ckpt.get("le_crop", None)
        le_stage = ckpt.get("le_stage", None)
        le_severity = ckpt.get("le_severity", None)
    else:
        model.load_state_dict(ckpt)
        le_crop = le_stage = le_severity = None

    model.to(DEVICE)
    model.eval()
    return model, le_crop, le_stage, le_severity

model, le_crop, le_stage, le_severity = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(input_tensor)
        crop_idx = torch.argmax(preds["crop"]).item()
        crop_name = le_crop.inverse_transform([crop_idx])[0]

        damaged_prob = torch.sigmoid(preds["is_damaged"]).item()
        damaged_label = "Yes" if damaged_prob > 0.5 else "No"

        growth_idx = torch.argmax(preds["growth"]).item()
        growth_name = le_stage.inverse_transform([growth_idx])[0]

        sev_idx = torch.argmax(preds["severity"]).item()
        sev_name = le_severity.inverse_transform([sev_idx])[0]

        health_score = preds["health"].item() * 100

    st.markdown("### 🧾 Prediction Results:")
    st.write(f"**Crop Type:** {crop_name}")
    st.write(f"**Damaged:** {damaged_label} ({damaged_prob:.2f})")
    st.write(f"**Growth Stage:** {growth_name}")
    st.write(f"**Severity Level:** {sev_name}")
    st.write(f"**Health Score:** {health_score:.2f}/100")
