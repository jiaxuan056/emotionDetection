import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ==============================
# 1. Load model and labels
# ==============================
model = load_model("model.h5")

# ⚠️ Make sure this order matches your training dataset!
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ==============================
# 2. Settings
# ==============================
CONF_THRESHOLD = 0.6     # ignore weak predictions
DISPLAY_THRESHOLD = 0.7  # show "Uncertain" if below this
QUEUE_SIZE = 30          # smoothing window

# Memory queue (store probabilities)
emotion_queue = deque(maxlen=QUEUE_SIZE)

# ==============================
# 3. Face detector
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# 4. Start webcam
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]

        # Preprocess
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_gray.astype('float32') / 255.0

        roi_expanded = np.expand_dims(roi_normalized, axis=0)
        roi_expanded = np.expand_dims(roi_expanded, axis=-1)

        # ==============================
        # 5. Predict
        # ==============================
        prediction = model.predict(roi_expanded, verbose=0)[0]

        # Ignore weak predictions
        if np.max(prediction) < CONF_THRESHOLD:
            continue

        # Store full probability vector
        emotion_queue.append(prediction)

        # ==============================
        # 6. Smooth prediction
        # ==============================
        avg_prediction = np.mean(emotion_queue, axis=0)
        max_index = np.argmax(avg_prediction)

        smoothed_emotion = emotion_labels[max_index]
        confidence = avg_prediction[max_index]

        # Apply display threshold
        if confidence < DISPLAY_THRESHOLD:
            smoothed_emotion = "Uncertain"

        # ==============================
        # 7. Draw results
        # ==============================
        label = f"{smoothed_emotion} ({confidence*100:.2f}%)"

        # Color coding
        if smoothed_emotion == 'Happy':
            color = (0, 255, 0)
        elif smoothed_emotion in ['Angry', 'Disgust', 'Fear']:
            color = (0, 0, 255)
        elif smoothed_emotion == "Uncertain":
            color = (128, 128, 128)
        else:
            color = (255, 255, 0)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2)

    # ==============================
    # 8. Show frame
    # ==============================
    cv2.imshow('Real-time Emotion Detection (Stable)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# 9. Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()