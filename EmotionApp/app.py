import streamlit as st
import os
import random
import numpy as np
import cv2
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.feature import hog, local_binary_pattern

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("😊 Emotion Detection System")

DATASET_PATH = "dataset"

emotion_labels = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# -------------------------
# LOAD MODELS (CACHE)
# -------------------------
@st.cache_resource
def load_models():
    cnn = load_model("models/best_CNNModel.keras")
    svm = joblib.load("models/svm_hog_lbp_model.joblib")
    return cnn, svm

cnn_model, svm_model = load_models()

# -------------------------
# SVM FUNCTIONS (IMPORTANT)
# -------------------------
def preprocess_rgb(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)

def lbp_spatial_histogram(gray, p=8, r=1, grid=4):
    lbp = local_binary_pattern(gray, p, r, method="uniform")
    n_bins = p + 2
    h, w = lbp.shape
    gh, gw = h // grid, w // grid

    hists = []
    for i in range(grid):
        for j in range(grid):
            patch = lbp[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            hist, _ = np.histogram(patch.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            hists.append(hist)

    return np.concatenate(hists)

def extract_features(gray):
    hog_vec = hog(gray,
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys')

    lbp_vec = lbp_spatial_histogram(gray)

    return np.concatenate([hog_vec, lbp_vec])

def predict_svm(model_dict, image):
    scaler = model_dict["scaler"]
    pca = model_dict["pca"]
    svc = model_dict["svc"]

    gray = preprocess_rgb(image)
    feat = extract_features(gray).reshape(1, -1)

    feat = scaler.transform(feat)
    feat = pca.transform(feat)

    prob = svc.predict_proba(feat)[0]

    return prob

# -------------------------
# RANDOM IMAGE FUNCTION
# -------------------------
def get_random_image():
    label = random.choice(emotion_labels)
    folder = os.path.join(DATASET_PATH, label)

    img_name = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, img_name)

    image = Image.open(img_path).convert("RGB")

    return image, label

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Select Model", ("CNN", "SVM"))

# -------------------------
# TAB 1: PREDICTION
# -------------------------
st.header("🔍 Random Emotion Prediction")

if st.button("🎲 Generate Random Image"):

    image, true_label = get_random_image()

    st.image(image, caption=f"Actual Emotion: {true_label}", use_column_width=True)

    # Convert to OpenCV
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # -------------------------
    # PREDICTION
    # -------------------------
    if model_choice == "CNN":
        face = cv2.resize(img, (100, 100))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = cnn_model.predict(face, verbose=0)[0]

    else:
        pred = predict_svm(svm_model, img)

    # -------------------------
    # RESULT
    # -------------------------
    idx = np.argmax(pred)
    pred_label = emotion_labels[idx]
    confidence = pred[idx] * 100

    st.subheader("📌 Prediction Result")

    if pred_label == true_label:
        st.success(f"✅ Correct! {pred_label} ({confidence:.2f}%)")
    else:
        st.error(f"❌ Wrong! Predicted: {pred_label} ({confidence:.2f}%) | Actual: {true_label}")

    # -------------------------
    # PROBABILITY BREAKDOWN
    # -------------------------
    st.write("### 📊 Confidence Breakdown")

    for i, emo in enumerate(emotion_labels):
        st.progress(int(pred[i] * 100))
        st.caption(f"{emo}: {pred[i]*100:.2f}%")