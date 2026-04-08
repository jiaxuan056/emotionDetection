import streamlit as st
import gdown
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image
from skimage.feature import hog

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
    if not os.path.exists(CNN_FILE_NAME):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)

    for file_id, file_name in zip(
        [SVM_MODEL_ID, SCALER_ID, ENCODER_ID],
        [SVM_MODEL_NAME, SCALER_NAME, ENCODER_NAME]
    ):
        if not os.path.exists(file_name):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_name, quiet=False)

    cnn_model = load_model(CNN_FILE_NAME)
    svm_model = joblib.load(SVM_MODEL_NAME)
    scaler = joblib.load(SCALER_NAME)
    label_encoder = joblib.load(ENCODER_NAME)

    return cnn_model, svm_model, scaler, label_encoder

cnn_model, svm_model, scaler, label_encoder = load_models()
st.success("✅ Models loaded successfully!")

st.title("😊 Emotion Detection Dashboard")

tab1, tab2 = st.tabs(["🖼️ Predict Emotion", "📊 Analysis & Reports"])

with tab1:
    st.markdown("Upload or select an image and detect emotion using CNN or HOG + SVM.")

    DATASET_PATH = "EmotionApp/dataset"
    emotion_category = st.selectbox("Select Emotion Folder", os.listdir(DATASET_PATH))
    category_path = os.path.join(DATASET_PATH, emotion_category)

    images_list = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    selected_image_name = st.selectbox("Select Image", images_list)
    selected_image_path = os.path.join(category_path, selected_image_name)

    image = Image.open(selected_image_path)
    st.image(image, caption=f"Selected Image: {selected_image_name}", width=300)

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

    if st.button("🚀 Predict Emotion"):

        col1, col2 = st.columns(2)

        # CNN
        with col1:
            st.markdown("### 🧠 CNN Model")
            input_cnn = preprocess_cnn(selected_image_path)
            pred_cnn = cnn_model.predict(input_cnn, verbose=0)[0]
            idx = np.argmax(pred_cnn)
            st.success(f"{EMOTION_LABELS[idx]} ({pred_cnn[idx]*100:.2f}%)")

            for i, em in enumerate(EMOTION_LABELS):
                st.metric(em, f"{pred_cnn[i]*100:.2f}%")
                st.progress(int(pred_cnn[i]*100))

        # SVM
        with col2:
            st.markdown("### 🧩 SVM Model")
            input_svm = preprocess_svm(selected_image_path)
            idx = svm_model.predict(input_svm)[0]
            prob = svm_model.predict_proba(input_svm)[0]

            st.success(f"{EMOTION_LABELS[idx]} ({prob[idx]*100:.2f}%)")

            for i, em in enumerate(EMOTION_LABELS):
                st.metric(em, f"{prob[i]*100:.2f}%")
                st.progress(int(prob[i]*100))

with tab2:
    st.subheader("📊 Model Performance & Reports")

    def download_image(file_id, filename):
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filename, quiet=False)
        return filename

    REPORT_IMAGES = {
        "train_test_distribution": "1e7m6yyHLGrolzW2_suH21j1-K78KBtnq",
        "cnn_confusion_matrix": "1m-A89sj3d-wYgisybaQ0ro3ayzrbK5sN",
        "cnn_classification_report": "1rMJshBnOjWaRBZTJsvEsi641jIm0g6iZ",
        "cnn_roc_curve": "17_tHKJ2Qfr_iBQCo9DxhD4ODvdmFqyZd",
        "cnn_training_curves": "1M2i3z9WCWofiqykqRyw0e6ifOd-wQ2GU",

        "svm_confusion_matrix": "1yfbRJ0RpyOlDOydmjYiTC2_7po6wMKxe",
        "svm_classification_report": "1EZ0tlzAE1_prShfmmzwPwDEXhgbArQSO",
        "svm_roc_curve": "1srjkWyUgdQuE0hHXYsVUNrvQqtHQWs50"
    }
    st.image(download_image(REPORT_IMAGES["train_test_distribution"], "test_distribution.png"), width=600)
    subtab1, subtab2 = st.tabs(["🧠 CNN", "🧩 SVM"])

    # CNN
    with subtab1:
        st.image(download_image(REPORT_IMAGES["cnn_training_curves"], "cnn_train.png"), width=600)
        st.image(download_image(REPORT_IMAGES["cnn_classification_report"], "cnn_report.png"), width=600)
        st.image(download_image(REPORT_IMAGES["cnn_confusion_matrix"], "cnn_cm.png"), width=600)
        st.image(download_image(REPORT_IMAGES["cnn_roc_curve"], "cnn_roc.png"), width=600)

    # SVM
    with subtab2:
        st.image(download_image(REPORT_IMAGES["svm_classification_report"], "svm_report.png"), width=600)
        st.image(download_image(REPORT_IMAGES["svm_confusion_matrix"], "svm_cm.png"), width=600)
        st.image(download_image(REPORT_IMAGES["svm_roc_curve"], "svm_roc.png"), width=600)
