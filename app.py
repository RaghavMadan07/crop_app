import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -----------------------
# Define your model (same as before)
# -----------------------
class CropNet(nn.Module):
    def __init__(self, num_classes=4, num_stages=3, num_severity=3):
        super(CropNet, self).__init__()
        self.backbone = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.crop_head = nn.Linear(in_features, num_classes)
        self.stage_head = nn.Linear(in_features, num_stages)
        self.severity_head = nn.Linear(in_features, num_severity)
        self.damage_head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        crop_class = self.crop_head(features)
        stage = self.stage_head(features)
        severity = self.severity_head(features)
        damage_prob = torch.sigmoid(self.damage_head(features))
        return crop_class, stage, severity, damage_prob


# -----------------------
# Load model with cache
# -----------------------
@st.cache_resource
def load_model():
    model = CropNet(num_classes=4, num_stages=3, num_severity=3)
    checkpoint = torch.load("clean_model.pth", map_location="cpu")

    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    st.write(f"✅ Model loaded with {len(checkpoint)} layers.")
    if missing:
        st.write(f"⚠️ Missing keys: {missing}")
    model.eval()
    return model

model = load_model()

# -----------------------
# Class labels for better UI
# -----------------------
CROP_CLASSES = ["Wheat", "Rice", "Maize", "Cotton"]
GROWTH_STAGES = ["Early Stage", "Mid Stage", "Mature Stage"]
SEVERITY_LEVELS = ["Healthy", "Moderate Infection", "Severe Infection"]

# -----------------------
# Streamlit UI
# -----------------------
st.title("🌾 Smart Crop Health Analyzer")
st.markdown("Upload a **crop field image** to analyze crop type, growth stage, and disease severity.")

uploaded_file = st.file_uploader("📸 Upload Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Crop Image", use_container_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)

    # Inference
    st.markdown("🔍 **Analyzing image...**")
    with torch.no_grad():
        crop_out, stage_out, severity_out, damage_prob = model(img_tensor)
        crop_idx = crop_out.argmax(1).item()
        stage_idx = stage_out.argmax(1).item()
        severity_idx = severity_out.argmax(1).item()
        damage_score = damage_prob.item()
        health_score = (1 - damage_score) * 100

    # Display
    st.success("✅ **Prediction complete!**")
    st.markdown("### 🧾 **Results:**")
    st.write(f"**Crop Type:** {CROP_CLASSES[crop_idx]}")
    st.write(f"**Growth Stage:** {GROWTH_STAGES[stage_idx]}")
    st.write(f"**Disease Severity:** {SEVERITY_LEVELS[severity_idx]}")
    st.write(f"**Damaged Probability:** {damage_score:.2f}")
    st.write(f"**Health Score:** {health_score:.2f}/100")
