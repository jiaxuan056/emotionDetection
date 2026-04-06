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
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# -------------------------
# 1️⃣ Model & Dataset Setup
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

DATASET_PATH = "EmotionApp/dataset"

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
# 2️⃣ App Layout Tabs
# -------------------------
tab1, tab2 = st.tabs(["🔍 Predict Emotion", "📊 Model Reports & Graphs"])

# -------------------------
# 3️⃣ Tab 1: Single Image Prediction
# -------------------------
with tab1:
    st.title("😊 Emotion Detection Dashboard")
    st.markdown("Upload or select an image from the dataset to predict emotion using CNN or SVM.")

    # Image selection
    emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
    category_path = os.path.join(DATASET_PATH, emotion_category)
    images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_image_name = st.selectbox("Select Image", images_list)
    selected_image_path = os.path.join(category_path, selected_image_name)
    image = Image.open(selected_image_path)
    st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)

    # Preprocessing
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

    # Prediction
    if st.button("🚀 Predict Emotion"):
        st.subheader("📌 Prediction Result")
        col1, col2 = st.columns(2)

        # CNN
        with col1:
            st.markdown("### 🧠 CNN Model")
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

        # SVM
        with col2:
            st.markdown("### 🧩 SVM Model")
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
# 4️⃣ Tab 2: Reports & Graphs
# -------------------------
with tab2:
    st.markdown("### 📊 Model Reports & Graphs")

    # Load test set predictions if precomputed
    # For demonstration purposes, we assume these files exist:
    # 'cnn_test_predictions.npz' and 'svm_test_predictions.npz'
    if os.path.exists("cnn_test_predictions.npz") and os.path.exists("svm_test_predictions.npz"):
        cnn_data = np.load("cnn_test_predictions.npz")
        X_test_cnn = cnn_data['X_test']
        y_test_enc_cnn = cnn_data['y_test']
        y_pred_cnn = cnn_data['y_pred']

        svm_data = np.load("svm_test_predictions.npz")
        X_test_svm = svm_data['X_test']
        y_test_enc_svm = svm_data['y_test']
        y_pred_svm = svm_data['y_pred']

        # Confusion matrices
        st.subheader("Confusion Matrices")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CNN Confusion Matrix**")
            cm_cnn = confusion_matrix(y_test_enc_cnn, y_pred_cnn)
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

        with col2:
            st.markdown("**SVM Confusion Matrix**")
            cm_svm = confusion_matrix(y_test_enc_svm, y_pred_svm)
            fig, ax = plt.subplots(figsize=(6,5))
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

        # Classification reports
        st.subheader("Classification Reports")
        st.markdown("**CNN Report**")
        st.text(classification_report(y_test_enc_cnn, y_pred_cnn, target_names=EMOTION_LABELS))
        st.markdown("**SVM Report**")
        st.text(classification_report(y_test_enc_svm, y_pred_svm, target_names=EMOTION_LABELS))

        # Accuracy metrics
        st.subheader("Accuracy Comparison")
        acc_cnn = accuracy_score(y_test_enc_cnn, y_pred_cnn)
        acc_svm = accuracy_score(y_test_enc_svm, y_pred_svm)
        st.metric("CNN Accuracy", f"{acc_cnn*100:.2f}%")
        st.metric("SVM Accuracy", f"{acc_svm*100:.2f}%")

        # Emotion distribution
        st.subheader("Emotion Distribution in Predictions")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CNN Predictions**")
            cnn_counts = pd.Series([EMOTION_LABELS[i] for i in y_pred_cnn]).value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=cnn_counts.index, y=cnn_counts.values, palette="Blues", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Emotion")
            st.pyplot(fig)

        with col2:
            st.markdown("**SVM Predictions**")
            svm_counts = pd.Series([EMOTION_LABELS[i] for i in y_pred_svm]).value_counts()
            fig, ax = plt.subplots(figsize=(6,4))
            sns.barplot(x=svm_counts.index, y=svm_counts.values, palette="Greens", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("Emotion")
            st.pyplot(fig)

    else:
        st.warning("Test prediction data not found. Please compute CNN & SVM test predictions first.")

st.info("💡 Tab 2 summarizes model performance: confusion matrices, classification reports, accuracy, and predicted emotion distributions.")
