# Enhanced Facial Emotion Detection with Music Playlist Playback
import cv2
from keras.models import model_from_json
import numpy as np
import webbrowser
import time
import random

# Load the model
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load Haar cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Labels and music links
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
music_links = {
    "sad": [
        "https://www.youtube.com/watch?v=2Vv-BfVoq4g",  # Ed Sheeran
        "https://www.youtube.com/watch?v=RgKAFK5djSk",  # Wiz Khalifa
        "https://www.youtube.com/watch?v=hLQl3WQQoQ0"   # Adele
    ],
    "happy": [
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",  # Pharrell Williams
        "https://www.youtube.com/watch?v=cmSbXsFE3l8",  # Meghan Trainor
        "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Mark Ronson
    ]
}

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Emotion detection function
def detect_emotion():
    webcam = cv2.VideoCapture(0)

    print("[INFO] Starting real-time emotion detection. Press 'q' to quit.")
    detected_once = False

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("[ERROR] Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = cv2.resize(face, (48, 48))
            face_input = extract_features(face)

            prediction = model.predict(face_input)
            label = labels[prediction.argmax()]

            cv2.putText(frame, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            print(f"[INFO] Detected emotion: {label}")

            if label in music_links and not detected_once:
                print(f"[ACTION] {label.capitalize()} detected. Launching music playlist...")
                detected_once = True
                webcam.release()
                cv2.destroyAllWindows()

                selected_songs = random.sample(music_links[label], min(2, len(music_links[label])))
                for link in selected_songs:
                    print(f"[PLAY] {link}")
                    webbrowser.open(link)
                    time.sleep(2)  # Let browser handle opening each link
                return

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[EXIT] User requested to quit.")
            break

    webcam.release()
    cv2.destroyAllWindows()

# Run the emotion detector
detect_emotion()
