import os

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


st.warning("⚠️ Only upload brain MRI images. Results may be unreliable for other images.")
st.title("Tumor Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

model_path = "best_model.pth"

@st.cache_resource
def load_model(path):
    # Build ResNet backbone and final layer (same as training)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    model = load_model(model_path)

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    label = "Tumor" if prob >= 0.5 else "No Tumor"

    st.markdown("**Prediction**")
    st.write(f"Result: **{label}**")
    st.write(f"Probability: {prob:.3f}")

    if not os.path.exists(model_path):
        st.warning(f"Model file '{model_path}' not found. Place the trained model in the project root.")
