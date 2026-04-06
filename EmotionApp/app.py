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
# 1️⃣ Download & Load Models
# -------------------------
CNN_FILE_ID = "1QVCR7tt7NFmZvMYkphjH3eOsVaaDgD7w"
SVM_FILE_ID = "1c94crk9dwSyabPI1neObqLZeNJTbo5_M"

CNN_FILE_NAME = "best_CNNModel.h5"
SVM_FILE_NAME = "svm_hog_lbp_model.joblib"

@st.cache_resource(show_spinner=True)
def load_models():
    # Download if missing
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)
    if not os.path.exists(SVM_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={SVM_FILE_ID}", SVM_FILE_NAME, quiet=False)
    
    # Load CNN
    cnn_model = load_model(CNN_FILE_NAME)
    
    # Load SVM dict
    svm_dict = joblib.load(SVM_FILE_NAME)
    scaler = svm_dict['scaler']
    pca = svm_dict['pca']
    svm_model = svm_dict['svc']       # ← Actual SVC object
    svm_classes = svm_dict['classes']
    svm_img_size = svm_dict['img_size']
    
    return cnn_model, (scaler, pca, svm_model, svm_classes, svm_img_size)

# Load models
cnn_model, svm_info = load_models()
scaler, pca, svm_model, svm_classes, SVM_IMG_SIZE = svm_info
st.success("✅ Models loaded successfully!")

# -------------------------
# 2️⃣ App Layout
# -------------------------
st.title("😊 Emotion Detection from Dataset Images")
st.markdown("Select an image from the dataset and detect emotion using CNN or SVM")

model_choice = st.selectbox("Select Model", ["CNN", "SVM"])

# -------------------------
# 3️⃣ Image Selection
# -------------------------
DATASET_PATH = "EmotionApp/dataset"  # replace with your dataset path
emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]

IMG_SIZE = 100 if model_choice == "CNN" else SVM_IMG_SIZE

# Let user select emotion category
emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
category_path = os.path.join(DATASET_PATH, emotion_category)

# List all images in folder
images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_name = st.selectbox("Select Image", images_list)
selected_image_path = os.path.join(category_path, selected_image_name)

# Show selected image
image = Image.open(selected_image_path)
st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)

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
    resized = cv2.resize(gray, (SVM_IMG_SIZE, SVM_IMG_SIZE))
    
    # HOG features
    features, _ = hog(resized,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      orientations=9,
                      visualize=True,
                      channel_axis=None)
    features = features.reshape(1, -1)
    
    # Scale + PCA
    features = scaler.transform(features)
    features = pca.transform(features)
    
    return features

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
        label = svm_classes[pred_idx]
        st.success(f"Predicted Emotion: {label}")
