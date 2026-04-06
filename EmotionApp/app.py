# app.py
import streamlit as st
import gdown
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image
from skimage.feature import hog

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
IMG_SIZE = 100 if model_choice == "CNN" else 128

# Let user select emotion category first
emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
category_path = os.path.join(DATASET_PATH, emotion_category)

# List all images in the selected folder
images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_name = st.selectbox("Select Image", images_list)
selected_image_path = os.path.join(category_path, selected_image_name)

# Show selected image with smaller width
image = Image.open(selected_image_path)
st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)  # 300 px width

# -------------------------
# 4️⃣ Prediction Functions
# -------------------------
def preprocess_image_cnn(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(resized.astype(np.float32)/255.0, axis=0)

def preprocess_image_svm(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # HOG features
    features, _ = hog(resized, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, channel_axis=None)
    return features.reshape(1, -1)

# -------------------------
# 5️⃣ Prediction Button
# -------------------------
if st.button("Predict Emotion"):
    if model_choice == "CNN":
        input_data = preprocess_image_cnn(selected_image_path)
        pred = cnn_model.predict(input_data, verbose=0)[0]
        max_idx = int(np.argmax(pred))
        label = emotion_labels[max_idx]
        confidence = pred[max_idx] * 100
        st.success(f"Predicted Emotion: {label} ({confidence:.2f}%)")
    else:
        input_features = preprocess_image_svm(selected_image_path)
        pred_idx = svm_model.predict(input_features)[0]
        label = emotion_labels[pred_idx]
        st.success(f"Predicted Emotion: {label}")
