import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from collections import deque

# ==============================
# Load model
# ==============================
model = load_model("model.h5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

CONF_THRESHOLD = 0.6
DISPLAY_THRESHOLD = 0.7
QUEUE_SIZE = 10

emotion_queue = deque(maxlen=QUEUE_SIZE)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# UI
# ==============================
st.title("😊 Emotion Detection (Cloud Version)")

img_file = st.camera_input("Take a photo")

if img_file is not None:
    # Convert image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi / 255.0

        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = model.predict(roi, verbose=0)[0]

        if np.max(prediction) < CONF_THRESHOLD:
            continue

        emotion_queue.append(prediction)

        avg_prediction = np.mean(emotion_queue, axis=0)
        max_index = np.argmax(avg_prediction)

        emotion = emotion_labels[max_index]
        confidence = avg_prediction[max_index]

        if confidence < DISPLAY_THRESHOLD:
            emotion = "Uncertain"

        label = f"{emotion} ({confidence*100:.2f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame)