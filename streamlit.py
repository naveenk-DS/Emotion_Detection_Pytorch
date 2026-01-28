# streamlit.py
# Real-Time Human Emotion Detection - Streamlit Interface

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os

# -------- CONFIG --------
MODEL_PATH = "model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")
st.title("😊 Real-Time Human Emotion Detection")

# -------- LOAD TRAINED MODEL --------
if not os.path.exists(MODEL_PATH):
    st.error("model.pth NOT FOUND. Please train the model first.")
    st.stop()

checkpoint = torch.load(MODEL_PATH, map_location=device)
class_names = checkpoint["class_names"]

from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=None)   # IMPORTANT
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.eval()


# -------- IMAGE PREPROCESSING --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# -------- FACE DETECTOR --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_emotion(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30))



    if len(faces) == 0:
        return None, "No face detected"

    x, y, w, h = faces[0]
    face = image.crop((x, y, x+w, y+h))

    image_tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, pred = torch.max(outputs, 1)
        emotion = class_names[pred.item()]

    return face, emotion

# -------- CAMERA INPUT --------
st.subheader("📷 Capture Image from Camera")
img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert("RGB")
    st.image(image, caption="Captured Image", width=300)

    face, result = predict_emotion(image)

    if face is None:
        st.error(result)
    else:
        st.image(face, caption="Detected Face", width=200)
        st.success(f"Predicted Emotion: **{result}**")

# -------- UPLOAD IMAGE --------
st.subheader("📁 Upload Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    face, result = predict_emotion(image)

    if face is None:
        st.error(result)
    else:
        st.image(face, caption="Detected Face", width=200)
        st.success(f"Predicted Emotion: **{result}**")
