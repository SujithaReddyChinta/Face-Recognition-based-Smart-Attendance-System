import face_recognition
import cv2
import numpy as np
import csv
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
import gudhi as gd 
from scipy.spatial import distance_matrix

yolo_model = YOLO("yolov8n-face-lindevs.pt")

image_dir = "photos"
encodings = {}
known_face_names = []

for fname in os.listdir(image_dir):
    file_path = os.path.join(image_dir, fname)
    image = face_recognition.load_image_file(file_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        encodings[fname] = encoding[0]
        known_face_names.append(fname.split('.')[0])  # Assuming the names are derived from file names

# Known face encodings and names
known_face_encodings = list(encodings.values())
known_face_names = [name.split('.')[0] for name in encodings.keys()]

def compute_persistent_homology(face_encodings):
    if len(face_encodings) < 2:
        return face_encodings  # Not enough points for homology

    # Compute distance matrix
    dist_matrix = distance_matrix(face_encodings, face_encodings)

    # Create Vietoris-Rips complex
    rips_complex = gd.RipsComplex(distance_matrix=dist_matrix, max_edge_length=1.0)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

    # Compute persistent homology
    persistence = simplex_tree.persistence()

    # Filter features with long persistence (stable features)
    stable_features = []
    for (dim, (birth, death)) in persistence:
        if death - birth > 0.1:  # Threshold for stability
            stable_features.append((birth, death))

    # If no stable features, return original encodings
    if not stable_features:
        return face_encodings

    # Use stable features to refine embeddings
    refined_encodings = np.mean(face_encodings, axis=0)  # Averaging stable features
    return [refined_encodings]
# Streamlit app
st.title("Face Recognition Attendance System")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read and process the uploaded image
    img = face_recognition.load_image_file(uploaded_image)

    # Create or open a CSV file for the current date
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    attendance_file = os.path.join(os.getcwd(), current_date + '.csv')

    # Initialize students list
    students = known_face_names.copy()
    attendance_records = []  # Store attendance records

    # Get YOLO results
    yolo_results = yolo_model(img)
    yolo_boxes = yolo_results[0].boxes

    # Find face locations and encodings for detected faces
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)
    
    face_names = []  # List to hold names for all detected faces

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default to "Unknown" for unmatched faces

        if True in matches:  # If there is a match
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]  # Get the name of the matched face

            # Attendance tracking
            if name in students:
                students.remove(name)
                # Record attendance in the list
                attendance_records.append([name, 'present'])
        face_names.append(name)  # Append the name to the list (either known or "Unknown")

    # Draw rectangles and labels for all detected faces
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the image in Streamlit
    st.image(img, caption='Processed Image', channels="RGB")

    # Write to CSV
    with open(attendance_file, 'w', newline='') as f:
        lnwriter = csv.writer(f)
        lnwriter.writerow(["Name", "Status"])  # Write header
        # Write present records
        lnwriter.writerows(attendance_records)
        # Write absent students
        for name in students:
            lnwriter.writerow([name, 'absent'])  