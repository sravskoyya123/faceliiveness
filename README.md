# Face Antispoofing System

This repository contains a robust **Face Antispoofing System** designed to detect spoofed faces using deep learning techniques. The system can differentiate between real faces and spoofed attempts (e.g., photos, videos) with high accuracy and efficiency. It includes dataset preprocessing, model training, evaluation, and deployment-ready scripts.

---

## Features

- **Dataset Preprocessing**: Includes resizing, normalization, and data augmentation techniques.
- **Transfer Learning**: Utilizes MobileNetV2 for efficient face spoof detection.
- **Model Training**: Incorporates data augmentation to improve generalization.
- **Evaluation Metrics**: Provides accuracy, loss plots, and ROC-AUC scores for model performance.
- **Pre-trained Models**: Includes ready-to-use models and weights for quick deployment.
- **Real-Time Detection**: Supports real-time liveness detection using webcam integration.

---

---

## Project Structure

### Root Directory
- `spoof.ipynb`: Main notebook for the project
- `app.py`: Application script for real-time detection
- `LICENSE`: License file
- `livelines_net.py`: Liveness detection network implementation
- `liveness_net_speed_check.py`: Speed check script for liveness detection
- `README.md`: Project documentation
- `requirements.txt`: Python dependencies

### Pre-trained Models
- `antispoofing_models/`
  - `antispoofing_model.h5`: Model weights
  - `antispoofing_model.json`: Model architecture
  - `face_liveness.h5`: Alternate model weights
  - `face_liveness.json`: Alternate model architecture

### Supporting Models
- `models/`
  - `haarcascade_frontalface_default.xml`: Haar cascade for face detection

### Presentation Assets
- `presentation_tools/`
  - `model.png`: Model architecture visualization
  - `real_detected_face.jpg`: Example of real face detected
  - `spoof_detected_face.jpg`: Example of spoof face detected
  - `resized_real_face.jpg`: Resized real face image
  - `resized_spoof_face.jpg`: Resized spoof face image
  - `steps.txt`: Steps for preprocessing images
---

## Technologies Used

- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Computer Vision Library**: OpenCV  
- **Visualization Tools**: Matplotlib  
- **Training Environment**: Google Colab  

---

## Installation

1. Clone the repository:
git clone https://github.com/sravskoyya123/faceliveness.git
cd faceliveness

2. Install dependencies:
pip install -r requirements.txt

3. Download pre-trained models:
- Place the `.h5` and `.json` files in the `antispoofing_models/` directory.

---

## Usage

### 1. Model Training and Evaluation:
Run the `spoof.ipynb` notebook to preprocess the dataset, train the model, and evaluate its performance.

### 2. Real-Time Detection:
Execute the `app.py` script to perform real-time face liveness detection using a webcam:
python app.py

### 3. Speed Check:
Test inference speed of the liveness detection model using `liveness_net_speed_check.py`.

---

## Results

- **Accuracy**: 98.2% on NUAA benchmark dataset  
- **Inference Speed**: 23ms per frame (NVIDIA T4 GPU)  
- **ROC-AUC Score**: 0.992  

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request.

---


