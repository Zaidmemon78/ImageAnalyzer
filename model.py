import torch
import torch.nn as nn
import torchvision.models as models

# YEH LINE HATA DO (Remove this line):
# from model import get_model  <--- DELETE THIS LINE (This was likely line 10)

def get_model(num_classes=2, pretrained=True):
    """Loads a pre-trained MobileNetV2 model and modifies the classifier."""

    # MobileNetV2 is lightweight, good for mobile and CPU
    model = models.mobilenet_v2(pretrained=pretrained)

    # Freeze all parameters in the feature extraction part
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Parameters of the new classifier layer requires_grad = True by default
    if __name__ != '__main__': # Print only when imported, not when run directly
        print(f"Loaded MobileNetV2. Replaced classifier for {num_classes} classes.")
        if pretrained:
            print("Feature extractor parameters frozen.")

    return model

# Yeh __main__ block test karne ke liye tha. Ismein 'get_model' direct call hoga, import ki zarurat nahi.
if __name__ == '__main__':
    # Test the model creation
    print("Testing model creation (running model.py directly)...")
    # Function isi file me define hai, toh direct call karo
    test_model = get_model(num_classes=2)
    print("Model created successfully.")
    # print(test_model) # Uncomment to see the model structure

    # Example input tensor
    dummy_input = torch.randn(1, 3, 224, 224)
    output = test_model(dummy_input)
    print("Dummy input shape:", dummy_input.shape)
    print("Output shape (batch_size, num_classes):", output.shape)