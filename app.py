import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import MultiHeadResNet  # Make sure this is your CNN model file

# ---------------------- CONFIG ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "clean_model.pth"

st.set_page_config(page_title="Crop Condition Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Crop Condition Prediction App")
st.write("Upload a crop image to predict its type, damage status, growth stage, severity, and health score.")

# ---------------------- MODEL LOADER ----------------------
@st.cache_resource
def load_model():
    model = MultiHeadResNet(
        backbone_name="resnet34",  # or whatever backbone you used in training
        pretrained=False
    ).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    
    # Handle both raw state_dict and dict with "model_state"
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ---------------------- LOAD MODEL ----------------------
model = load_model()

# ---------------------- IMAGE TRANSFORMS ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Upload a crop image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with st.spinner("🔍 Analyzing image..."):
        with torch.no_grad():
            preds = model(input_tensor)

            crop_idx = torch.argmax(preds["crop"]).item()
            growth_idx = torch.argmax(preds["growth"]).item()
            sev_idx = torch.argmax(preds["severity"]).item()
            damaged_prob = torch.sigmoid(preds["is_damaged"]).item()
            damaged_label = "Yes" if damaged_prob > 0.5 else "No"
            health_score = preds["health"].item() * 100

    # ---------------------- DISPLAY RESULTS ----------------------
    st.markdown("### 🧾 Prediction Results")
    st.success(f"**Crop Type (index):** {crop_idx}")
    st.write(f"**Damaged:** {damaged_label} ({damaged_prob:.2f})")
    st.write(f"**Growth Stage (index):** {growth_idx}")
    st.write(f"**Severity Level (index):** {sev_idx}")
    st.write(f"**Health Score:** {health_score:.2f}/100")

else:
    st.info("👆 Please upload a crop image to get predictions.")
