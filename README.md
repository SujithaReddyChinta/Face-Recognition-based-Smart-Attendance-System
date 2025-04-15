# TDA-Practicum
# Overview

This project implements a face recognition-based attendance system using YOLOv8 and face_recognition. It supports two modes:

- face-recognition.py: A Streamlit-based web application for image-based attendance.

- smart-attendance.py: A real-time video-based attendance system using OpenCV.

# Features

- Face Detection with YOLOv8: Uses a YOLOv8 face detection model for precise face localization.

- Face Recognition with face_recognition: Compares detected faces with stored images to mark attendance.

- Persistent Homology for Feature Stability: Uses Gudhi to filter stable facial features.

- Streamlit-based Web UI: Allows image uploads for attendance marking.

- CSV-based Attendance Tracking: Logs attendance records for each session.

# Requirements

- Before running the project, install the required dependencies, run:

 pip install face-recognition opencv-python numpy pandas streamlit ultralytics gudhi scipy

 # YOLO Model

Ensure you have the YOLOv8 face detection model in the project directory:

face-recognition.py uses yolov8n-face-lindevs.pt

smart-attendance.py uses yolov8/yolov8n-face.pt

Download YOLO models from Ultralytics YOLOv8.

# How to Use( Instructions)

- First clone the repo in VS Code:    git clone <repo-url>

- Navigate to your folder:  cd <your-project-folder>

- Install requirements:  pip install -r requirements.txt
   You might come across an error with dlib. I installed it locally. Note: dlib only works on python version 3.9 and 3.10, it wouldn't with versions higher than that.

- Install other libraries: pip install -U face-recognition face_recognition_models opencv-python streamlit numpy scipy pandas matplotlib seaborn gudhi ultralytics

- Finally run this command and Run the Streamlit application using this command:  streamlit run face-recognition.py

- Upload an image containing faces. The system will recognize and mark attendance in a CSV file (date-based).

# Output Format

Attendance is logged in a CSV file named with the current date (YYYY-MM-DD.csv)
