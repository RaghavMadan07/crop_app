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

uploaded_file = st.file_uploader("Upload an image of a crop", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and show the image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    input_tensor = transform(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

    st.write("🔍 Running model inference...")
    with torch.no_grad():
        preds = model(input_tensor)

        # Apply softmax/sigmoid to get readable outputs
        crop_probs = torch.softmax(preds["crop"], dim=1)
        growth_probs = torch.softmax(preds["growth"], dim=1)
        severity_probs = torch.softmax(preds["severity"], dim=1)
        damage_prob = torch.sigmoid(preds["is_damaged"]).item()
        health_score = torch.sigmoid(preds["health"]).item() * 100

        # Convert to readable outputs
        crop_idx = torch.argmax(crop_probs).item()
        growth_idx = torch.argmax(growth_probs).item()
        severity_idx = torch.argmax(severity_probs).item()

    # -----------------------
    # Display predictions
    # -----------------------
    st.markdown("### 🧾 Prediction Results")
    st.write(f"**Crop Class Index:** {crop_idx}")
    st.write(f"**Growth Stage Index:** {growth_idx}")
    st.write(f"**Severity Level Index:** {severity_idx}")
    st.write(f"**Damaged Probability:** {damage_prob:.2f}")
    st.write(f"**Health Score:** {health_score:.2f}/100")

    st.success("✅ Prediction complete!")

else:
    st.info("Please upload an image to begin analysis.")
