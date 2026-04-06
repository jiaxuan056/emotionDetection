# app.py
import streamlit as st
import gdown
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque
from PIL import Image

# -------------------------
# 1️⃣ Download Models from Google Drive
# -------------------------
CNN_FILE_ID = "1QVCR7tt7NFmZvMYkphjH3eOsVaaDgD7w"
SVM_FILE_ID = "1c94crk9dwSyabPI1neObqLZeNJTbo5_M"

CNN_FILE_NAME = "best_CNNModel.h5"  # Changed to .h5
SVM_FILE_NAME = "svm_hog_lbp_model.joblib"

@st.cache_resource(show_spinner=True)
def load_models():
    # Download CNN if not exists
    if not st.session_state.get('cnn_downloaded', False):
        gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", CNN_FILE_NAME, quiet=False)
        st.session_state['cnn_downloaded'] = True

    # Download SVM if not exists
    if not st.session_state.get('svm_downloaded', False):
        gdown.download(f"https://drive.google.com/uc?id={SVM_FILE_ID}", SVM_FILE_NAME, quiet=False)
        st.session_state['svm_downloaded'] = True

    # Load models
    cnn_model = load_model(CNN_FILE_NAME)
    svm_model = joblib.load(SVM_FILE_NAME)
    return cnn_model, svm_model

cnn_model, svm_model = load_models()
st.success("✅ Models loaded successfully!")

# -------------------------
# 2️⃣ App Layout
# -------------------------
st.title("😊 Real-time Emotion Detection")
st.markdown("Detect emotions from your webcam using CNN or SVM")

model_choice = st.selectbox("Select Model", ["CNN", "SVM"])

# -------------------------
# 3️⃣ Webcam Stream using Streamlit Camera Input
# -------------------------
emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
IMG_SIZE = 100 if model_choice=="CNN" else 128
emotion_queue = deque(maxlen=20)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if model_choice=="CNN":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized.astype(np.float32)/255.0, axis=0)
    else:
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        return resized

# Streamlit camera input
stframe = st.empty()
camera_image = st.camera_input("📷 Use your webcam")

if camera_image:
    img = np.array(Image.open(camera_image))
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5, minSize=(48,48))
    for (x,y,w,h) in faces:
        roi = img[y:y+h, x:x+w]
        if model_choice=="CNN":
            batch = preprocess_frame(roi)
            pred = cnn_model.predict(batch, verbose=0)[0]
            emotion_queue.append(pred)
            avg_pred = np.mean(emotion_queue, axis=0)
            max_idx = int(np.argmax(avg_pred))
            label = f"{emotion_labels[max_idx]} ({avg_pred[max_idx]*100:.1f}%)"
        else:
            label = "SVM demo"  # placeholder for SVM

        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

    stframe.image(img, channels="BGR")
