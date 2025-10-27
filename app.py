import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import MultiHeadResNet

# ---------------------
# CONFIGURATION
# ---------------------
MODEL_PATH = "clean_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Class label mappings
CROP_LABELS = [
    'Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon',
    'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana',
    'cardamom', 'chilli', 'clove', 'coconut', 'cotton', 'gram', 'jowar', 'jute',
    'maize', 'mustard-oil', 'papaya', 'pineapple', 'rice', 'soyabean', 'sugarcane',
    'sunflower', 'tea', 'tomato', 'vigna-radiati(Mung)', 'wheat'
]
GROWTH_LABELS = ['early', 'late', 'mid']
SEVERITY_LABELS = ['healthy', 'mild', 'moderate', 'severe']

# ---------------------
# MODEL LOADING
# ---------------------
@st.cache_resource
def load_model():
    st.write("🔄 Loading model...")
    model = MultiHeadResNet(backbone_name="resnet50", pretrained=False,
                            num_crops=len(CROP_LABELS),
                            num_stages=len(GROWTH_LABELS),
                            num_severity=len(SEVERITY_LABELS))
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    st.success(f"✅ Model loaded successfully with {len(state_dict)} layers.")
    if missing:
        st.warning(f"Missing keys: {missing}")
    if unexpected:
        st.info(f"Unexpected keys: {unexpected}")
        
    model.to(DEVICE).eval()
    return model

model = load_model()

# ---------------------
# IMAGE TRANSFORM
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------
# STREAMLIT UI
# ---------------------
st.title("🌱 Crop Condition Classification App")
st.markdown("Upload a crop image to get predictions for **crop type, growth stage, damage severity, and health**.")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Run Prediction"):
        st.info("Running model inference...")

        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            output = model(input_tensor)

            # Interpret outputs
            crop_idx = torch.argmax(output["crop"]).item()
            growth_idx = torch.argmax(output["growth"]).item()
            severity_idx = torch.argmax(output["severity"]).item()
            damaged_prob = torch.sigmoid(output["is_damaged"]).item()
            health_score = max(0, min(100, 100 * torch.sigmoid(output["health"]).item()))

            # Get readable labels
            crop_label = CROP_LABELS[crop_idx]
            growth_label = GROWTH_LABELS[growth_idx]
            severity_label = SEVERITY_LABELS[severity_idx]

        st.subheader("🧾 Prediction Results")
        st.write(f"**Crop Type:** {crop_label}")
        st.write(f"**Growth Stage:** {growth_label}")
        st.write(f"**Severity Level:** {severity_label}")
        st.write(f"**Damaged Probability:** {damaged_prob:.2f}")
        st.write(f"**Health Score:** {health_score:.2f} / 100")
        st.success("✅ Prediction complete!")

else:
    st.info("Please upload an image to begin.")

