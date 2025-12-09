import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# Pehle 'yolov8n.pt' tha (Nano - Fast but less accurate)
# Ab 'yolov8s.pt' use kar rahe hain (Small - Thoda better accuracy)
# Agar laptop achha hai to 'yolov8m.pt' (Medium) bhi try kar sakte ho
print("Model load ho raha hai... (Thoda time lag sakta hai)")
model = YOLO('yolov8s.pt')

# --- CAMERA START ---
cap = cv2.VideoCapture(0)

# Window ka naam
window_name = "Super Smart Object Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

print("\n--- INSTRUCTIONS ---")
print("1. Camera ke saamne objects (Phone, Bottle, Cup) laao.")
print("2. Object ko thoda STABLE rakho (hilana mat).")
print("3. Band karne ke liye Video Window par CLICK karo aur 'q' dabao.")
print("--------------------\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera access failed!")
        break

    # --- DETECTION ---
    # conf=0.3 ka matlab: Agar 30% bhi lagta hai ki bottle hai, to dikha do.
    # Isse chhoti aur door ki cheezein bhi detect hongi.
    results = model(frame, conf=0.3, verbose=False)

    # Draw boxes
    annotated_frame = results[0].plot()

    # Show
    cv2.imshow(window_name, annotated_frame)

    # Exit Logic
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()