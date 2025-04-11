import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tqdm import tqdm
import time

root_dir = os.getcwd()
# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# Load Anti-Spoofing Model graph
json_file = open('antispoofing_models/face_liveness.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('antispoofing_models/face_liveness.h5')
print("Model loaded from disk")

# This functionality has been moved to the Flask application in app.py

