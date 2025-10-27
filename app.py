import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from model import MultiHeadResNet  # Make sure this is your CNN model file

# ---------------------- CONFIG ----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "clean_model.pth"
BACKBONE = "resnet50"  # ✅ define this

st.set_page_config(page_title="Crop Condition Prediction", page_icon="🌾", layout="centered")
st.title("🌾 Crop Condition Prediction App")
st.write("Upload a crop image to predict its type, damage status, growth stage, severity, and health score.")

# ---------------------- MODEL LOADER ----------------------

@st.cache_resource
def load_model():
    BACKBONE = "resnet18"  # change if you used a different one
    model = MultiHeadResNet(backbone_name=BACKBONE, pretrained=False).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    new_state_dict = model.state_dict()

    # ✅ Handle all possible checkpoint types
    if isinstance(ckpt, dict):
        # case 1: has "model_state"
        if "model_state" in ckpt:
            ckpt = ckpt["model_state"]

        # case 2: keys start with "module."
        if any(k.startswith("module.") for k in ckpt.keys()):
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        # case 3: keys start with "backbone."
        if any(k.startswith("backbone.") for k in ckpt.keys()):
            for k, v in ckpt.items():
                if k in new_state_dict:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Loaded backbone weights only.")
            model.eval()
            return model

        # case 4: direct match
        try:
            model.load_state_dict(ckpt, strict=False)
            print("✅ Loaded checkpoint with partial match.")
        except RuntimeError as e:
            print(f"⚠️ Partial load error ignored: {e}")
            for k, v in ckpt.items():
                if k in new_state_dict and new_state_dict[k].shape == v.shape:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Loaded compatible weights only.")
    else:
        print("⚠️ Unexpected checkpoint type, loading skipped.")

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
