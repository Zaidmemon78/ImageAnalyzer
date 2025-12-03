import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# ImageNet statistics for normalization (important for pre-trained models)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Define transformations for training and validation data
# Training includes augmentation (random changes) to make the model robust
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224), # Common input size for many models
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Validation transform is simpler, just resize and normalize
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# --- Data directory structure assumption ---
# data/
#   train/
#     human/
#       img1.jpg
#       img2.jpg
#       ...
#     no_human/
#       img100.jpg
#       img101.jpg
#       ...
#   val/
#     human/
#       img500.jpg
#       ...
#     no_human/
#       img600.jpg
#       ...
# -----------------------------------------

DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')

def get_dataloaders(batch_size=32):
    """Creates training and validation dataloaders."""
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
        print(f"Error: Please create '{TRAIN_DIR}' and '{VAL_DIR}' directories")
        print("Expected structure:")
        print("data/")
        print("  train/")
        print("    human/")
        print("    no_human/")
        print("  val/")
        print("    human/")
        print("    no_human/")
        return None, None, None

    train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(VAL_DIR, transform=val_transform)

    # Check if datasets are empty
    if len(train_dataset) == 0:
        print(f"Error: No images found in {TRAIN_DIR}. Please add images.")
        return None, None, None
    if len(val_dataset) == 0:
        print(f"Warning: No images found in {VAL_DIR}. Validation will be skipped.")
        # You might want to handle this more gracefully, maybe exit or proceed without validation

    print(f"Found {len(train_dataset)} training images belonging to {len(train_dataset.classes)} classes: {train_dataset.class_to_idx}")
    print(f"Found {len(val_dataset)} validation images belonging to {len(val_dataset.classes)} classes: {val_dataset.class_to_idx}")

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("Warning: Train and Validation class indices do not match!")
        print(f"Train: {train_dataset.class_to_idx}")
        print(f"Val:   {val_dataset.class_to_idx}")


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) # num_workers > 0 can speed up loading if CPU has cores
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Store class names for later use (e.g., in prediction)
    class_names = train_dataset.classes

    return train_loader, val_loader, class_names

if __name__ == '__main__':
    # Test the dataloader
    print("Testing DataLoader...")
    train_loader, val_loader, class_names = get_dataloaders(batch_size=4)
    if train_loader and val_loader:
        print("DataLoader created successfully.")
        # Get a sample batch
        try:
            images, labels = next(iter(train_loader))
            print("Sample batch shape:", images.shape)
            print("Sample labels:", labels)
            print("Class names:", class_names)
        except StopIteration:
            print("Could not get a batch from train_loader (is it empty?)")
    else:
        print("Failed to create DataLoaders. Check data directory structure and contents.")