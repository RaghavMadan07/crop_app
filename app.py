import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import plotly.graph_objects as go
import torch.nn.functional as F
from model import MultiHeadResNet

# ----------------------------
# 🔧 PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="🌾 Crop Health Detector", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f6fff8;
            color: #102a43;
        }
        h1, h2, h3 {
            color: #206a5d;
        }
        .stMetric {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🧠 LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = MultiHeadResNet(backbone_name="resnet50", pretrained=False, num_crops=30, num_stages=3, num_severity=4)
    checkpoint = torch.load("model.pth", map_location="cpu")
    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    st.write(f"✅ Model loaded successfully. Missing keys: {missing}")
    model.eval()
    return model

model = load_model()

# ----------------------------
# 📋 LABELS
# ----------------------------
crop_labels = [
    'Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon',
    'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana',
    'cardamom', 'chilli', 'clove', 'coconut', 'cotton', 'gram', 'jowar',
    'jute', 'maize', 'mustard-oil', 'papaya', 'pineapple', 'rice',
    'soyabean', 'sugarcane', 'sunflower', 'tea', 'tomato',
    'vigna-radiati(Mung)', 'wheat'
]
growth_labels = ['early', 'late', 'mid']
severity_labels = ['healthy', 'mild', 'moderate', 'severe']

# ----------------------------
# 🖼️ SIDEBAR INFO
# ----------------------------
with st.sidebar:
    st.title("📘 About the Model")
    st.write("This AI model identifies crop type, growth stage, damage severity, and health score from crop images.")
    st.markdown("**Model:** Multi-Head ResNet50")
    st.markdown("**Classes:** 30 Crops | 3 Growth Stages | 4 Severity Levels")
    st.divider()
    st.info("💡 Tip: Upload clear daylight crop images for best results!")

# ----------------------------
# 📸 IMAGE UPLOAD
# ----------------------------
st.title("🌾 AI-Powered Crop Health Detection")
uploaded_file = st.file_uploader("Upload an image of your crop", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(img, caption="Uploaded Crop Image", use_container_width=True)

    # ----------------------------
    # 🔄 PREPROCESS & INFERENCE
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)

    crop_idx = torch.argmax(output["crop"]).item()
    growth_idx = torch.argmax(output["growth"]).item()
    severity_idx = torch.argmax(output["severity"]).item()
    damaged_prob = torch.sigmoid(output["is_damaged"]).item()
    health_score = (1 - damaged_prob) * 100 if severity_labels[severity_idx] == "healthy" else (1 - damaged_prob * 0.8) * 100

    crop_name = crop_labels[crop_idx]
    growth_stage = growth_labels[growth_idx]
    severity_label = severity_labels[severity_idx]

    with col2:
        st.markdown("### 🧾 Prediction Summary")
        st.metric("🌱 Crop Type", crop_name)
        st.metric("🪴 Growth Stage", growth_stage)
        st.metric("⚠️ Severity Level", severity_label)
        st.metric("💧 Damage Probability", f"{damaged_prob:.2f}")
        st.metric("❤️ Health Score", f"{health_score:.2f}/100")

        # Display health gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health_score,
            title={'text': "Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if health_score > 70 else "orange" if health_score > 40 else "red"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffb3b3"},
                    {'range': [40, 70], 'color': "#ffe680"},
                    {'range': [70, 100], 'color': "#b3ffb3"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Severity message
        if severity_label == "healthy":
            st.success("✅ Crop is healthy and thriving!")
        elif severity_label == "mild":
            st.info("🌤 Mild issues detected — monitor conditions.")
        elif severity_label == "moderate":
            st.warning("⚠️ Moderate stress — consider early treatment.")
        else:
            st.error("🚨 Severe damage detected! Immediate attention needed.")

# ----------------------------
# 👣 FOOTER
# ----------------------------
st.markdown("""
---
👨‍💻 **Developed by:** Your Name  
🚀 **Powered by:** Streamlit & PyTorch  
🌱 **Model:** Multi-Head ResNet50 | v1.0.0
""")
