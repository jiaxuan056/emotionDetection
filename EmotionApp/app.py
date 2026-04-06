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
import matplotlib.pyplot as plt
import seaborn as sns

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

IMG_SIZE_CNN = 100
IMG_SIZE_SVM = 100

EMOTION_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

@st.cache_resource(show_spinner=True)
def load_models():
    # Download CNN
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)

    # Download SVM files
    for file_id, file_name in zip([SVM_MODEL_ID, SCALER_ID, ENCODER_ID],
                                  [SVM_MODEL_NAME, SCALER_NAME, ENCODER_NAME]):
        if not os.path.exists(file_name):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

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
st.title("😊 Emotion Detection Dashboard")
st.markdown(
    "Upload or select an image from the dataset and detect emotion using CNN or HOG + SVM."
)

# Tabs for UI
tab1, tab2 = st.tabs(["🖼️ Predict Emotion", "📊 Analysis & Reports"])

DATASET_PATH = "EmotionApp/dataset"

# -------------------------
# 3️⃣ Tab 1: Prediction
# -------------------------
with tab1:
    st.subheader("Select Image for Prediction")
    emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
    category_path = os.path.join(DATASET_PATH, emotion_category)

    images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_image_name = st.selectbox("Select Image", images_list)
    selected_image_path = os.path.join(category_path, selected_image_name)

    image = Image.open(selected_image_path)
    st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)

    # Preprocessing functions
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
        features_scaled = scaler.transform([features])
        return features_scaled

    if st.button("🚀 Predict Emotion"):

        st.subheader("📌 Prediction Result")

        col1, col2 = st.columns(2)

        # CNN Prediction
        with col1:
            st.markdown("### 🧠 CNN Model Prediction")
            input_cnn = preprocess_cnn(selected_image_path)
            pred_cnn = cnn_model.predict(input_cnn, verbose=0)[0]
            idx_cnn = np.argmax(pred_cnn)
            label_cnn = EMOTION_LABELS[idx_cnn]
            confidence_cnn = pred_cnn[idx_cnn] * 100
            st.success(f"Predicted Emotion: {label_cnn} ({confidence_cnn:.2f}%)")

            st.write("#### Emotion Probabilities (CNN)")
            for i, em in enumerate(EMOTION_LABELS):
                st.metric(em, f"{pred_cnn[i]*100:.2f}%")
                st.progress(int(pred_cnn[i]*100))

        # SVM Prediction
        with col2:
            st.markdown("### 🧩 SVM Model Prediction")
            input_svm = preprocess_svm(selected_image_path)
            pred_svm_idx = svm_model.predict(input_svm)[0]
            pred_svm_prob = svm_model.predict_proba(input_svm)[0]
            label_svm = EMOTION_LABELS[pred_svm_idx]
            confidence_svm = pred_svm_prob[pred_svm_idx] * 100
            st.success(f"Predicted Emotion: {label_svm} ({confidence_svm:.2f}%)")

            st.write("#### Emotion Probabilities (SVM)")
            for i, em in enumerate(EMOTION_LABELS):
                st.metric(em, f"{pred_svm_prob[i]*100:.2f}%")
                st.progress(int(pred_svm_prob[i]*100))

        st.divider()
        st.info("💡 Both models show prediction probabilities. Compare CNN vs SVM confidence levels.")

# -------------------------
# 4️⃣ Tab 2: Analysis & Reports (CNN + SVM)
# -------------------------
# Mapping numeric folder to labels
NUM_FOLDER_TO_LABEL = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral"
}

# Google Drive path
BASE = "Mydrive/AI Assignment/trainKaggle"  # replace with actual mounted path if needed

with tab2:
    st.subheader("📊 Model Performance & Confusion Matrices")

    test_images, test_labels = [], []

    for folder_name, label_name in NUM_FOLDER_TO_LABEL.items():
        folder_path = os.path.join(BASE, folder_name)
        if not os.path.exists(folder_path):
            st.warning(f"Folder {folder_path} not found!")
            continue
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                test_images.append(img_path)
                test_labels.append(EMOTION_LABELS.index(label_name))

    st.write(f"Total images loaded for evaluation: {len(test_images)}")

    if len(test_images) == 0:
        st.warning("⚠️ No images found. Check your folder structure!")
    else:
        y_true = np.array(test_labels)

        # ---------------- CNN Predictions ----------------
        X_cnn = np.array([preprocess_cnn(p)[0] for p in test_images])
        y_pred_cnn_probs = cnn_model.predict(X_cnn, verbose=0)
        y_pred_cnn = np.argmax(y_pred_cnn_probs, axis=1)
        acc_cnn = accuracy_score(y_true, y_pred_cnn)
        st.metric("CNN Accuracy", f"{acc_cnn*100:.2f}%")

        conf_mat_cnn = confusion_matrix(y_true, y_pred_cnn)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(conf_mat_cnn, annot=True, fmt='d', cmap='Blues',
                    xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("CNN Confusion Matrix")
        st.pyplot(fig)

        # ---------------- SVM Predictions ----------------
        X_svm = []
        for img_path in test_images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE_SVM, IMG_SIZE_SVM))
            features = hog(resized, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           orientations=9, block_norm='L2-Hys')
            X_svm.append(features)
        X_svm = scaler.transform(X_svm)
        y_pred_svm = svm_model.predict(X_svm)
        acc_svm = accuracy_score(y_true, y_pred_svm)
        st.metric("SVM Accuracy", f"{acc_svm*100:.2f}%")

        conf_mat_svm = confusion_matrix(y_true, y_pred_svm)
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(conf_mat_svm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("SVM Confusion Matrix")
        st.pyplot(fig)
