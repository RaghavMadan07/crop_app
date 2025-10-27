import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MultiHeadResNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = MultiHeadResNet(
        backbone_name="resnet50",
        pretrained=False,
        num_crops=5,
        num_stages=3,
        num_severity=4
    ).to(DEVICE)

    ckpt = torch.load("clean_model.pth", map_location=DEVICE)
    state_dict = ckpt["model_state"]

    # Remove possible 'module.' prefixes from multi-GPU training
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load state dict (non-strict to handle minor mismatches)
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    st.write("✅ Model loaded successfully!")
    if missing:
        st.warning(f"Missing keys: {missing}")
    if unexpected:
        st.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    return model

# Load the model once
model = load_model()

# ----------------------- Streamlit UI -----------------------

st.title("🌾 Crop Condition Classification App")
st.write("Upload a crop image to predict its class, damage, growth stage, severity, and health score.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)

    # Convert outputs to displayable values
    st.subheader("🧠 Model Predictions")
    st.write({
        "Crop Type (argmax)": torch.argmax(outputs["crop"], dim=1).item(),
        "Is Damaged (sigmoid > 0.5)": (torch.sigmoid(outputs["is_damaged"]) > 0.5).item(),
        "Growth Stage (argmax)": torch.argmax(outputs["growth"], dim=1).item(),
        "Severity Level (argmax)": torch.argmax(outputs["severity"], dim=1).item(),
        "Health Score (sigmoid)": torch.sigmoid(outputs["health"]).item()
    })
