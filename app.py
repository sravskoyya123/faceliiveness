from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import os

app = Flask(__name__)

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/face_liveness.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('antispoofing_models/face_liveness.h5')
print("Model loaded from disk")

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img_arr = np.fromstring(file.read(), np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []

    for (x, y, w, h) in faces:
        face = img[y-5:y+h+5, x-5:x+w+5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)
        preds = model.predict(resized_face)[0]
        label = 'spoof' if preds > 0.5 else 'real'
        results.append({'label': label, 'confidence': float(preds)})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
