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
SVM_MODEL_ID = "1jpJ2HwviM6PX91RHX81GjQynu_Pqy-Cm"
SCALER_ID = "1llZiJ8z-uDJE7L9UB6xv3nqe4irKHPBN"
ENCODER_ID = "1AWE9JbAl-Z7DtrwDV4oCRWOtjvOstFMU"

CNN_FILE_NAME = "best_CNNModel.h5"
SVM_MODEL_NAME = "svm_emotion_model.pkl"
SCALER_NAME = "scaler.pkl"
ENCODER_NAME = "label_encoder.pkl"

EMOTION_LABELS = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

@st.cache_resource(show_spinner=True)
def load_models():
    # Download CNN
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)

    # Download SVM files
    if not os.path.exists(SVM_MODEL_NAME):
        gdown.download(f"https://drive.google.com/uc?id={SVM_MODEL_ID}", SVM_MODEL_NAME, quiet=False)

    if not os.path.exists(SCALER_NAME):
        gdown.download(f"https://drive.google.com/uc?id={SCALER_ID}", SCALER_NAME, quiet=False)

    if not os.path.exists(ENCODER_NAME):
        gdown.download(f"https://drive.google.com/uc?id={ENCODER_ID}", ENCODER_NAME, quiet=False)

    # Load models
    cnn_model = load_model(CNN_FILE_NAME)
    svm_model = joblib.load(SVM_MODEL_NAME)
    scaler = joblib.load(SCALER_NAME)
    label_encoder = joblib.load(ENCODER_NAME)

    return cnn_model, svm_model, scaler, label_encoder

cnn_model, svm_model, scaler, label_encoder = load_models()
st.success("✅ Models loaded successfully!")

# -------------------------
# 2️⃣ App Layout
# -------------------------
st.title("😊 Emotion Detection")
st.markdown("Detect emotion using CNN or HOG + SVM")

model_choice = st.selectbox("Select Model", ["CNN", "SVM"])

# -------------------------
# 3️⃣ Image Selection
# -------------------------
DATASET_PATH = "EmotionApp/dataset"

IMG_SIZE_CNN = 100
IMG_SIZE_SVM = 100

emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
category_path = os.path.join(DATASET_PATH, emotion_category)

images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image_name = st.selectbox("Select Image", images_list)
selected_image_path = os.path.join(category_path, selected_image_name)

image = Image.open(selected_image_path)
st.image(image, caption=f"{selected_image_name}", width=300)

st.divider()

# -------------------------
# 4️⃣ Preprocessing Functions
# -------------------------
def preprocess_cnn(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE_CNN, IMG_SIZE_CNN))
    return np.expand_dims(resized.astype(np.float32)/255.0, axis=0)

def preprocess_svm(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE_SVM, IMG_SIZE_SVM))
    features = hog(resized,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   orientations=9,
                   block_norm='L2-Hys')
    return scaler.transform([features])

# -------------------------
# 5️⃣ Prediction & Display
# -------------------------
if st.button("🚀 Predict Emotion"):
    
    st.subheader("📌 Prediction Result")

    if model_choice == "CNN":
        input_data = preprocess_cnn(selected_image_path)
        pred = cnn_model.predict(input_data, verbose=0)[0]
        idx = np.argmax(pred) + 1        # 1-based label
        label = EMOTION_LABELS[idx - 1] # map to emotion name
        confidence = pred[idx - 1] * 100

        st.success(f"Predicted Emotion: {label} ({confidence:.2f}%)")

        # METRICS
        colA, colB = st.columns(2)
        colA.metric("Confidence (%)", f"{confidence:.2f}")
        colB.metric("Other Emotions (%)", f"{100 - confidence:.2f}")

        # PROGRESS BAR
        st.write("### 📊 Confidence Visualization")
        st.progress(int(confidence))
        st.caption(f"{label} Confidence")

    else:
        features = preprocess_svm(selected_image_path)
        pred_idx = svm_model.predict(features)[0]
        pred_prob = svm_model.predict_proba(features)[0]
        label = EMOTION_LABELS[pred_idx]
        confidence = pred_prob[pred_idx] * 100

        st.success(f"Predicted Emotion: {label} ({confidence:.2f}%)")

        # METRICS
        colA, colB = st.columns(2)
        colA.metric("Confidence (%)", f"{confidence:.2f}")
        colB.metric("Other Emotions (%)", f"{100 - confidence:.2f}")

        # PROGRESS BAR
        st.write("### 📊 Confidence Visualization")
        st.progress(int(confidence))
        st.caption(f"{label} Confidence")
