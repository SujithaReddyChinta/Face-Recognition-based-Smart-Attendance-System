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
