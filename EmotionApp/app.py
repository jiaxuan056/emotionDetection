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
TRAIN_FOLDER_ID = "1uIHJ8vWI3KDMChvWnFZK3ZfxtAV_Y8Tf"

REPORT_IMAGES = {
    "train_data_distribution": "1aKu4b1CKQYSHG58F_mLpYXQ3640WsuuB",
    "train_test_distribution": "1e7m6yyHLGrolzW2_suH21j1-K78KBtnq",
    "cnn_classification_report": "1rMJshBnOjWaRBZTJsvEsi641jIm0g6iZ",
    "cnn_confusion_matrix": "1m7ObYYpsvQssSFThAtUQshxnYYy99ZlY",
    "cnn_roc_curve": "1_R_EdEFpf7bKkMat_Sr3iIHSTlTsG3cm",
    "cnn_training_curves": "1M2i3z9WCWofiqykqRyw0e6ifOd-wQ2GU",
    "svm_classification_report": "1EZ0tlzAE1_prShfmmzwPwDEXhgbArQSO",
    "svm_confusion_matrix": "1yfbRJ0RpyOlDOydmjYiTC2_7po6wMKxe",
    "svm_roc_curve": "1srjkWyUgdQuE0hHXYsVUNrvQqtHQWs50"
}

CNN_FILE_NAME = "best_CNNModel.h5"
SVM_MODEL_NAME = "svm_emotion_model.pkl"
SCALER_NAME = "scaler.pkl"
ENCODER_NAME = "label_encoder.pkl"
TRAIN_FOLDER_NAME = "trainKaggle"

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

    @st.cache_resource(show_spinner=True)
    def download_train_images():
        if not os.path.exists(TRAIN_FOLDER_NAME):
            st.info("Downloading training images from Google Drive...")
            gdown.download_folder(f"https://drive.google.com/drive/folders/{TRAIN_FOLDER_ID}", output=TRAIN_FOLDER_NAME, quiet=False)
        return TRAIN_FOLDER_NAME
    
    BASE = download_train_images()

    def download_image(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

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
with tab2:
    st.subheader("📊 Model Performance & Reports")

    st.markdown("### 📂 Dataset Distribution")

    col1, col2 = st.columns(2)

    with col1:
        img_path = download_image(REPORT_IMAGES["train_data_distribution"], "train_data.png")
        st.image(img_path, caption="Training Data Distribution", use_column_width=True)

    with col2:
        img_path = download_image(REPORT_IMAGES["train_test_distribution"], "train_test.png")
        st.image(img_path, caption="Train vs Test Split", use_column_width=True)

    st.divider()

    # ---------------- CNN ----------------
    st.markdown("## 🧠 CNN Model Reports")

    col1, col2 = st.columns(2)

    with col1:
        img_path = download_image(REPORT_IMAGES["cnn_confusion_matrix"], "cnn_cm.png")
        st.image(img_path, caption="CNN Confusion Matrix", use_column_width=True)

    with col2:
        img_path = download_image(REPORT_IMAGES["cnn_classification_report"], "cnn_report.png")
        st.image(img_path, caption="CNN Classification Report", use_column_width=True)

    col3, col4 = st.columns(2)

    with col3:
        img_path = download_image(REPORT_IMAGES["cnn_roc_curve"], "cnn_roc.png")
        st.image(img_path, caption="CNN ROC Curve", use_column_width=True)

    with col4:
        img_path = download_image(REPORT_IMAGES["cnn_training_curves"], "cnn_train.png")
        st.image(img_path, caption="CNN Training Curves", use_column_width=True)

    st.divider()

    # ---------------- SVM ----------------
    st.markdown("## 🧩 SVM Model Reports")

    col5, col6 = st.columns(2)

    with col5:
        img_path = download_image(REPORT_IMAGES["svm_confusion_matrix"], "svm_cm.png")
        st.image(img_path, caption="SVM Confusion Matrix", use_column_width=True)

    with col6:
        img_path = download_image(REPORT_IMAGES["svm_classification_report"], "svm_report.png")
        st.image(img_path, caption="SVM Classification Report", use_column_width=True)

    col7 = st.columns(1)[0]

    with col7:
        img_path = download_image(REPORT_IMAGES["svm_roc_curve"], "svm_roc.png")
        st.image(img_path, caption="SVM ROC Curve", use_column_width=True)

    st.info("💡 These are pre-generated evaluation results from your trained models.")
