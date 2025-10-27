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
import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# -----------------------
# User Interface for Prediction
# -----------------------

st.subheader("📸 Upload a Crop Image")

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



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