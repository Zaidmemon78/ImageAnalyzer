import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import get_model  # Tumhari model file se

# --- CONFIGURATION ---
MODEL_PATH = 'models/human_detector_mobilenetv2.pth'
CLASS_NAMES = ['human', 'no_human']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD MODEL ---
print("Loading Model...")
model = get_model(num_classes=2, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model Loaded! Starting Camera...")

# --- PREPROCESSING ---
# Wahi same transform jo training me use kiya tha
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- CAMERA START ---
cap = cv2.VideoCapture(0) # 0 matlab default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCV image (BGR) ko PIL Image (RGB) mein badlo
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Model ke liye taiyaar karo
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    # Predict karo
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    class_name = CLASS_NAMES[predicted_idx.item()]
    conf_score = confidence.item()

    # --- RESULT DIKHAO ---
    # Agar Human hai to GREEN, nahi to RED
    color = (0, 255, 0) if class_name == 'human' else (0, 0, 255)
    text = f"{class_name}: {conf_score:.2f}"

    # Frame pe text likho
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (5, 5), (300, 60), (0,0,0), 2) # Text ke liye box

    # Screen par dikhao
    cv2.imshow('AI Human Detector (Press Q to Exit)', frame)

    # 'Q' dabane par band ho jayega
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()