import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Import model architecture
from model import MultiHeadResNet

# -----------------------
# ⚙️ Basic Streamlit Setup
# -----------------------
st.set_page_config(page_title="Crop Condition Classifier", layout="wide")
st.title("🌾 CROPIC: Multi-Task Crop Health and Damage Detection")
st.write("Upload a crop image to predict crop type, damage, growth stage, severity, and overall health.")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "clean_model.pth"

# -----------------------
# 🧠 Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = MultiHeadResNet(backbone_name="resnet50", pretrained=False,
                            num_crops=30, num_stages=3, num_severity=4)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Handle mismatched keys safely
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------
# 🖼️ Image Upload Section
# -----------------------
uploaded_file = st.file_uploader("Upload a crop image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------
    # 🔄 Preprocessing
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -----------------------
    # 🔍 Inference
    # -----------------------
    with torch.no_grad():
        outputs = model(input_tensor)
        crop_probs = F.softmax(outputs["crop"], dim=1)
        growth_probs = F.softmax(outputs["growth"], dim=1)
        severity_probs = F.softmax(outputs["severity"], dim=1)
        damaged_prob = torch.sigmoid(outputs["is_damaged"])
        health_score = torch.sigmoid(outputs["health"])

        crop_pred = torch.argmax(crop_probs, dim=1).item()
        growth_pred = torch.argmax(growth_probs, dim=1).item()
        severity_pred = torch.argmax(severity_probs, dim=1).item()

    # -----------------------
    # 📊 Display Results
    # -----------------------
    st.subheader("🧾 Predicted Results:")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Predicted Crop ID:** {crop_pred}")
        st.write(f"**Damage Probability:** {damaged_prob.item():.2f}")
        st.write(f"**Health Score (0-1):** {health_score.item():.2f}")

    with col2:
        st.write(f"**Growth Stage (ID):** {growth_pred}")
        st.write(f"**Severity Level (ID):** {severity_pred}")

    st.success("✅ Prediction completed successfully!")

else:
    st.info("Please upload a crop image to begin inference.")
