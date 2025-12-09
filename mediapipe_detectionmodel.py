import cv2
import mediapipe as mp
import numpy as np
import os
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
MODEL_FILE = 'efficientdet_lite0.tflite'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'

# --- 1. DOWNLOAD MODEL (Agar nahi hai) ---
if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE} from Google...")
    response = requests.get(MODEL_URL)
    with open(MODEL_FILE, 'wb') as f:
        f.write(response.content)
    print("Download Complete!")

# --- 2. SETUP MEDIAPIPE ---
base_options = python.BaseOptions(model_asset_path=MODEL_FILE)
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)  # 50% sure hone par hi dikhana
detector = vision.ObjectDetector.create_from_options(options)


# --- VISUALIZATION FUNCTION (Boxes banane ke liye) ---
def draw_landmarks(image, detection_result):
    for detection in detection_result.detections:
        # Box banao
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

        # Category (Naam) nikalo
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)

        # Color Decide karo (Person=Green, Baaki=Red)
        color = (0, 255, 0) if category_name == 'person' else (0, 0, 255)

        # Draw on Image
        cv2.rectangle(image, start_point, end_point, color, 3)

        # Text likho
        result_text = f"{category_name} ({int(probability * 100)}%)"
        cv2.putText(image, result_text, (bbox.origin_x, bbox.origin_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


# --- 3. CAMERA LOOP ---
cap = cv2.VideoCapture(0)
cv2.namedWindow('Ultimate AI Detector', cv2.WINDOW_NORMAL)

print("System Ready! Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe ko RGB image chahiye hoti hai
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # DETECT KARO!
    detection_result = detector.detect(mp_image)

    # Result dikhao
    annotated_image = draw_landmarks(frame, detection_result)

    cv2.imshow('Ultimate AI Detector', annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()