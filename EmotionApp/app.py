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
# 1️⃣ Google Drive IDs
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

# -------------------------
# 2️⃣ File Names
# -------------------------
CNN_FILE_NAME = "best_CNNModel.h5"
SVM_MODEL_NAME = "svm_emotion_model.pkl"
SCALER_NAME = "scaler.pkl"
ENCODER_NAME = "label_encoder.pkl"
TRAIN_FOLDER_NAME = "trainKaggle"

IMG_SIZE_CNN = 100
IMG_SIZE_SVM = 100

EMOTION_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# -------------------------
# 3️⃣ Download Image Function (FIXED)
# -------------------------
def download_image(file_id, filename):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
    return filename

# -------------------------
# 4️⃣ Load Models
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models():

    # CNN
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME)

    # SVM + tools
    for file_id, file_name in zip(
        [SVM_MODEL_ID, SCALER_ID, ENCODER_ID],
        [SVM_MODEL_NAME, SCALER_NAME, ENCODER_NAME]
    ):
        if not os.path.exists(file_name):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name)

    # Dataset folder
    if not os.path.exists(TRAIN_FOLDER_NAME):
        st.info("Downloading dataset...")
        gdown.download_folder(
            f"https://drive.google.com/drive/folders/{TRAIN_FOLDER_ID}",
            output=TRAIN_FOLDER_NAME
        )

    cnn_model = load_model(CNN_FILE_NAME)
    svm_model = joblib.load(SVM_MODEL_NAME)
    scaler = joblib.load(SCALER_NAME)
    label_encoder = joblib.load(ENCODER_NAME)

    return cnn_model, svm_model, scaler, label_encoder

cnn_model, svm_model, scaler, label_encoder = load_models()
st.success("✅ Models loaded successfully!")

# -------------------------
# 5️⃣ UI Layout
# -------------------------
st.title("😊 Emotion Detection Dashboard")
st.markdown("Upload or select an image to detect emotion using CNN or HOG + SVM.")

tab1, tab2 = st.tabs(["🖼️ Predict Emotion", "📊 Analysis & Reports"])

DATASET_PATH = "EmotionApp/dataset"

# -------------------------
# 6️⃣ TAB 1: Prediction
# -------------------------
with tab1:
    st.subheader("Select Image")

    emotion_category = st.selectbox("Emotion Folder", os.listdir(DATASET_PATH))
    category_path = os.path.join(DATASET_PATH, emotion_category)

    images_list = [f for f in os.listdir(category_path) if f.endswith(('png','jpg','jpeg'))]
    selected_image_name = st.selectbox("Image", images_list)
    selected_image_path = os.path.join(category_path, selected_image_name)

    image = Image.open(selected_image_path)
    st.image(image, width=300)

    def preprocess_cnn(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE_CNN, IMG_SIZE_CNN))
        return np.expand_dims(img / 255.0, axis=0)

    def preprocess_svm(path):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (IMG_SIZE_SVM, IMG_SIZE_SVM))
        features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9)
        return scaler.transform([features])

    if st.button("🚀 Predict"):

        col1, col2 = st.columns(2)

        # CNN
        with col1:
            st.markdown("### 🧠 CNN")
            pred = cnn_model.predict(preprocess_cnn(selected_image_path))[0]
            idx = np.argmax(pred)
            st.success(f"{EMOTION_LABELS[idx]} ({pred[idx]*100:.2f}%)")

        # SVM
        with col2:
            st.markdown("### 🧩 SVM")
            pred = svm_model.predict(preprocess_svm(selected_image_path))[0]
            prob = svm_model.predict_proba(preprocess_svm(selected_image_path))[0]
            st.success(f"{EMOTION_LABELS[pred]} ({prob[pred]*100:.2f}%)")

# -------------------------
# 7️⃣ TAB 2: Reports (IMAGES)
# -------------------------
with tab2:
    st.subheader("📊 Model Reports")

    st.markdown("### 📂 Dataset Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.image(download_image(REPORT_IMAGES["train_data_distribution"], "train.png"))

    with col2:
        st.image(download_image(REPORT_IMAGES["train_test_distribution"], "split.png"))

    st.divider()

    # CNN
    st.markdown("## 🧠 CNN Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(download_image(REPORT_IMAGES["cnn_confusion_matrix"], "cnn_cm.png"))

    with col2:
        st.image(download_image(REPORT_IMAGES["cnn_classification_report"], "cnn_report.png"))

    st.image(download_image(REPORT_IMAGES["cnn_roc_curve"], "cnn_roc.png"))
    st.image(download_image(REPORT_IMAGES["cnn_training_curves"], "cnn_train.png"))

    st.divider()

    # SVM
    st.markdown("## 🧩 SVM Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(download_image(REPORT_IMAGES["svm_confusion_matrix"], "svm_cm.png"))

    with col2:
        st.image(download_image(REPORT_IMAGES["svm_classification_report"], "svm_report.png"))

    st.image(download_image(REPORT_IMAGES["svm_roc_curve"], "svm_roc.png"))

    st.info("💡 Pre-generated evaluation results from your trained models.")
