# app.py
import streamlit as st
import gdown
import joblib
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from collections import deque

# -------------------------
# 1️⃣ Download Models from Google Drive
# -------------------------
CNN_FILE_ID = "1QVCR7tt7NFmZvMYkphjH3eOsVaaDgD7w"
SVM_FILE_ID = "1c94crk9dwSyabPI1neObqLZeNJTbo5_M"

@st.cache_data(show_spinner=True)
def download_models():
    gdown.download(f"https://drive.google.com/uc?id={CNN_FILE_ID}", "best_CNNModel.keras", quiet=False)
    gdown.download(f"https://drive.google.com/uc?id={SVM_FILE_ID}", "svm_hog_lbp_model.joblib", quiet=False)
    cnn_model = load_model("best_CNNModel.keras")
    svm_model = joblib.load("svm_hog_lbp_model.joblib")
    return cnn_model, svm_model

cnn_model, svm_model = download_models()
st.success("✅ Models loaded successfully!")

# -------------------------
# 2️⃣ App Layout
# -------------------------
st.title("😊 Real-time Emotion Detection")
st.markdown("Detect emotions from your webcam using CNN or SVM")

model_choice = st.selectbox("Select Model", ["CNN", "SVM"])

# -------------------------
# 3️⃣ Webcam Stream
# -------------------------
emotion_labels = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Angry", "Neutral"]
IMG_SIZE = 100 if model_choice=="CNN" else 128
CONF_THRESHOLD = 0.35
emotion_queue = deque(maxlen=20)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if model_choice=="CNN":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized.astype(np.float32)/255.0, axis=0)
    else:  # SVM HOG+LBP
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        return resized

# Streamlit camera
frame_window = st.image([])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot open webcam")
else:
    st.info("Webcam running... press ESC to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5, minSize=(48,48))
        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            if model_choice=="CNN":
                batch = preprocess_frame(roi)
                pred = cnn_model.predict(batch, verbose=0)[0]
                emotion_queue.append(pred)
                avg_pred = np.mean(emotion_queue, axis=0)
                max_idx = int(np.argmax(avg_pred))
                label = f"{emotion_labels[max_idx]} ({avg_pred[max_idx]*100:.1f}%)"
            else:
                # Preprocess for SVM
                gray = preprocess_frame(roi)
                # Extract HOG + LBP features as in your svm script
                # For simplicity, assume svm_features(gray) returns proper feature vector
                # features = svm_features(gray)
                # pred_idx = svm_model.predict([features])[0]
                # label = emotion_labels[pred_idx]
                label = "SVM demo"  # placeholder

            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()
