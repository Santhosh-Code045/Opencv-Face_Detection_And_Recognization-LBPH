# Face Detection and Recognition using LBPH & OpenCV

This project implements a custom face recognition pipeline using **OpenCV** and the **LBPH (Local Binary Patterns Histograms)** algorithm. It features a modular approach to detect faces in static images, process live video, and train a model on custom datasets for identification.



## 🛠 Features
* **Dual Detection:** Separate paths for testing detection on static images and video files.
* **LBPH Implementation:** Uses Local Binary Patterns for robust recognition.
* **Automated Training:** Scans a directory of labeled folders to generate a recognition model.
* **Visual Output:** Identifies individuals and overlays their names on grayscale processed frames.

## 📋 Prerequisites
The project is built using **Python 3.10**. Ensure you have the following packages installed:

```bash
pip install opencv-python opencv-contrib-python numpy
📂 Project Structure

    face_detection.py: Contains logic for initial face detection.

    face_train.py: Trains the recognizer and generates the model files.

    face_recognization.py: The final inference script for identifying faces.

    face_detection.xml: Haar Cascade classifier (from opencv/data/haarcascades).
🚀 How to Use
1. Initial Face Detection

Use face_detection.py to verify your Haar Cascade setup.

    Path 1: Set this to the path of your test image.

    Path 2: Set this to the path of your test video.

2. Prepare the Training Data

Organize your images into a main directory (referred to as Path 3).

    Important: Inside this directory, create a separate folder for each person. Ensure there are no loose files or non-person folders in Path 3.
Dataset/ (Path 3)
├── Person_A/
│   ├── img1.jpg
│   └── img2.jpg
└── Person_B/
    ├── img1.jpg
    └── img2.jpg
3. Training the Model

Run face_train.py.

    Use Path 3 to point to your dataset.

    This script will generate a .yml file and two .npy files (features and labels) in your directory.

4. Recognition (Inference)

Run face_recognization.py to identify a person. You will need to configure the following:

    Path 3: The directory of your training data (used for label mapping).

    Path 4: The path to the generated .yml training file.

    Path 5: The path to the specific image you want the model to identify.

The script will detect the face, compare it against the model, and print the person's name on a grayscale version of the image.
⚙️ Built With

    Python 3.10

    OpenCV (cv2): Main framework for computer vision.

    NumPy: For high-performance array operations.

    OS Module: For directory traversal and file handling.
