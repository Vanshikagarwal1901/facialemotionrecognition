from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from keras.models import model_from_json

app = Flask(__name__)
CORS(app)  # allow CORS so your frontend can call this API

# Load model
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels and music
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
    "sad": "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
    "happy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs"
}

def preprocess_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 48, 48, 1)
    return reshaped

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # image sent as base64 string data:image/jpeg;base64,/9j/...
    img_data = data['image']
    # Strip off the header (data:image/jpeg;base64,)
    img_str = img_data.split(',')[1]

    # Decode base64 string to bytes
    img_bytes = base64.b64decode(img_str)

    # Convert bytes to np array, then decode to OpenCV image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detect faces
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return jsonify({"emotion": None})

    # For simplicity, take first detected face
    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]

    processed_face = preprocess_face(face_img)
    prediction = model.predict(processed_face)
    emotion_idx = prediction.argmax()
    emotion = labels[emotion_idx]

    response = {"emotion": emotion}
    if emotion in music_links:
        response["music"] = music_links[emotion]

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
