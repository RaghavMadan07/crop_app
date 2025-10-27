import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Define model class (must match your Kaggle architecture)
from your_model_file import YourCNNModel  # if you defined it in another file

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YourCNNModel()
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# Define transforms (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # adjust to your model’s input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

st.title("🌾 Crop Condition Classification App")
st.write("Upload a crop image to get predictions for crop type, damage, growth, severity, and health!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img_t = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds = model(img_t)

        # Example: adjust based on your model output structure
        crop_pred = torch.argmax(preds["crop"], 1).item()
        is_damaged = torch.sigmoid(preds["is_damaged"]).item() > 0.5
        growth_pred = torch.argmax(preds["growth"], 1).item()
        severity_pred = torch.argmax(preds["severity"], 1).item()
        health_pred = F.softmax(preds["health"], dim=1).cpu().numpy().tolist()

        st.subheader("🧾 Predictions")
        st.write(f"**Crop Type:** {crop_pred}")
        st.write(f"**Damaged:** {'Yes' if is_damaged else 'No'}")
        st.write(f"**Growth Stage:** {growth_pred}")
        st.write(f"**Severity:** {severity_pred}")
        st.write(f"**Health Scores:** {health_pred}")
