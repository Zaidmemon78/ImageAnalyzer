import torch
from torchvision import transforms
from PIL import Image
import argparse
import os

from model import get_model # Reuse the model definition

# --- Configuration ---
MODEL_PATH = 'models/human_detector_mobilenetv2.pth'
NUM_CLASSES = 2 # Must match the trained model
CLASS_NAMES = ['human', 'no_human'] # IMPORTANT: Must match the order in training data (alphabetical usually by ImageFolder)
                                    # Check output of data_loader.py or train.py to confirm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same transformations as validation (without random augmentation)
predict_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, num_classes):
    """Loads the trained model state."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    # Initialize the model architecture
    model = get_model(num_classes=num_classes, pretrained=False) # Don't load pretrained weights again

    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE)) # map_location ensures it loads correctly even if trained on GPU and predicting on CPU
        model.eval() # Set model to evaluation mode
        model = model.to(DEVICE)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in model.py matches the saved model.")
        return None


def predict_image(image_path, model, class_names):
    """Makes a prediction on a single image file."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        # Load and transform the image
        image = Image.open(image_path).convert('RGB') # Ensure image is RGB
        image_tensor = predict_transform(image)
        # Add batch dimension (model expects batch of images)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        # Make prediction
        with torch.no_grad(): # No need to track gradients for prediction
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = class_names[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score

    except Exception as e:
        print(f"Error processing image or predicting: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict if an image contains a human.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Load the trained model
    loaded_model = load_model(MODEL_PATH, NUM_CLASSES)

    if loaded_model:
        # Make prediction
        prediction, confidence = predict_image(args.image, loaded_model, CLASS_NAMES)

        if prediction:
            print(f"\nPrediction for: {args.image}")
            print(f"Predicted class: {prediction}")
            print(f"Confidence: {confidence:.4f}")