# app.py
import streamlit as st
import gdown
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image
from collections import deque

# -------------------------
# 1️⃣ Download Models from Google Drive
# -------------------------
CNN_FILE_ID = "1QVCR7tt7NFmZvMYkphjH3eOsVaaDgD7w"
SVM_FILE_ID = "1GWQR8acH4uycRAAupBAOVtLMnEtzBrfK"

CNN_FILE_NAME = "best_CNNModel.h5"
SVM_FILE_NAME = "svm_emotion_model.joblib"

@st.cache_resource(show_spinner=True)
def load_models():
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)
    if not os.path.exists(SVM_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={SVM_FILE_ID}", SVM_FILE_NAME, quiet=False)
    
    cnn_model = load_model(CNN_FILE_NAME)
    svm_model = joblib.load(SVM_FILE_NAME)
    return cnn_model, svm_model

cnn_model, svm_model = load_models()
st.success("✅ Models loaded successfully!")

# -------------------------
# 2️⃣ App Layout
# -------------------------
st.title("😊 Emotion Detection from Dataset Images")
st.markdown("Select an image from the dataset and detect emotion using CNN or SVM")

model_choice = st.selectbox("Select Model", ["CNN", "SVM"])

# -------------------------
# 3️⃣ Image Selection from Dataset
# -------------------------
DATASET_PATH = "EmotionApp/dataset"  # replace with your dataset path
emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
IMG_SIZE = 100 if model_choice=="CNN" else 128

# Let user select emotion category first
emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
category_path = os.path.join(DATASET_PATH, emotion_category)

# List all images in the selected folder
images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_name = st.selectbox("Select Image", images_list)
selected_image_path = os.path.join(category_path, selected_image_name)

# Show selected image
image = Image.open(selected_image_path)
# Show selected image with smaller width
st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)  # 300 px width

# -------------------------
# 4️⃣ Prediction Function
# -------------------------
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if model_choice == "CNN":
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized.astype(np.float32)/255.0, axis=0)
    else:
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        return resized  # placeholder for SVM

if st.button("Predict Emotion"):
    input_data = preprocess_image(selected_image_path)
    if model_choice == "CNN":
        pred = cnn_model.predict(input_data, verbose=0)[0]
        max_idx = int(np.argmax(pred))
        label = emotion_labels[max_idx]
        confidence = pred[max_idx] * 100
        st.success(f"Predicted Emotion: {label} ({confidence:.2f}%)")
    else:
        st.info("SVM prediction placeholder")  # You can add HOG+LBP extraction here later
