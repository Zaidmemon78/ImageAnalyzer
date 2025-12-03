import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import copy

# Assuming data_loader.py and model.py are in the same directory
from data_loader import get_dataloaders
from model import get_model

# --- Configuration ---
DATA_DIR = 'data' # Folder name for your dataset
MODEL_SAVE_DIR = 'models' # Folder where the trained model will be saved
NUM_CLASSES = 2 # human, no_human (jitni categories hain tumhare data folder me)
BATCH_SIZE = 16 # Chota rakho agar CPU pe memory kam hai ya training slow hai. Adjust kar sakte ho.
NUM_EPOCHS = 10 # Kitni baar poore dataset pe train karna hai. Shuruat ke liye kam rakho.
LEARNING_RATE = 0.001 # Model kitni tezi se seekhega
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Automatic detection, CPU use karega agar GPU nahi

print(f"Using device: {DEVICE}") # Batayega ki CPU use ho raha hai ya GPU

# Create models directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# File path jahan model save hoga
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'human_detector_mobilenetv2.pth')

# --- Main Training Function ---
def train_model():
    print("Loading data...")
    # data_loader.py se training aur validation data load karo
    train_loader, val_loader, class_names = get_dataloaders(batch_size=BATCH_SIZE)

    # Agar data load nahi hua (e.g., folder nahi mila) toh exit karo
    if not train_loader:
        print("Failed to get dataloaders. Exiting training.")
        print(f"Make sure your data is inside the '{DATA_DIR}' folder")
        print("Expected structure: data/train/human, data/train/no_human, etc.")
        return

    # Check karo ki data loader se mili classes NUM_CLASSES se match karti hain ya nahi
    if class_names and len(class_names) != NUM_CLASSES:
         print(f"Error: DataLoader found {len(class_names)} classes ({class_names}), but NUM_CLASSES is set to {NUM_CLASSES}.")
         print("Please update NUM_CLASSES in train.py to match the number of folders inside data/train.")
         return

    print(f"Classes found: {class_names}") # Print karo konsi classes mili

    print("Loading model...")
    # model.py se pre-trained model load karo
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE) # Model ko CPU/GPU pe bhejo

    # Loss function define karo (classification ke liye CrossEntropyLoss standard hai)
    criterion = nn.CrossEntropyLoss()

    # Optimizer define karo (Adam ek common aur acha optimizer hai)
    # Hum sirf nayi classification layer ke parameters ko optimize kar rahe hain (Transfer Learning)
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=LEARNING_RATE)

    # Optional: Agar poora model train karna hai (slow hoga, zyada data chahiye) toh neeche wali line uncomment karo
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # print("Optimizer will train ALL model parameters.")

    # Optional: Learning rate kam karne ke liye scheduler (abhi ke liye zaruri nahi)
    # from torch.optim.lr_scheduler import StepLR
    # scheduler = StepLR(optimizer, step_size=7, gamma=0.1) # Har 7 epoch baad LR ko 0.1 se multiply karo

    print("Starting training...")
    since = time.time() # Training start time note karo

    best_model_wts = copy.deepcopy(model.state_dict()) # Best model ke weights store karne ke liye
    best_acc = 0.0 # Best validation accuracy store karne ke liye

    # Training loop - har epoch ke liye
    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Har epoch mein ek training phase aur ek validation phase hota hai
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Model ko training mode mein set karo (dropout etc. enable ho jayega)
                dataloader = train_loader
            else:
                # Agar validation data nahi hai toh validation phase skip karo
                if val_loader is None:
                    print("Validation loader not available, skipping validation phase.")
                    continue
                model.eval()   # Model ko evaluation mode mein set karo (dropout etc. disable ho jayega)
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Dataloader se batches mein data lo
            batch_count = 0
            for inputs, labels in dataloader:
                batch_count += 1
                # Data ko CPU/GPU pe bhejo
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Optimizer ke gradients ko zero karo har batch se pehle
                optimizer.zero_grad()

                # Forward pass: Input ko model se pass karo
                # Sirf training phase mein gradients calculate karo
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # Model prediction dega (logits)
                    _, preds = torch.max(outputs, 1) # Sabse zyada value wale index ko prediction maano
                    loss = criterion(outputs, labels) # Loss calculate karo

                    # Backward pass + optimize: Sirf training phase mein
                    if phase == 'train':
                        loss.backward() # Gradients calculate karo
                        optimizer.step() # Model weights update karo

                # Statistics calculate karo (loss aur accuracy)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                # Optional: Print progress within epoch
                if batch_count % 20 == 0: # Har 20 batch ke baad print karo
                   print(f'Epoch {epoch+1} [{phase}] Batch {batch_count}/{len(dataloader)} Loss: {loss.item():.4f}')


            # Epoch ke end mein scheduler step karo (agar use kar rahe ho)
            # if phase == 'train':
            #      if scheduler: scheduler.step()

            # Poore epoch ka average loss aur accuracy calculate karo
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Agar validation phase hai aur accuracy pehle se behtar hai, toh model save karo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Best model ko file mein save karo
                torch.save(best_model_wts, MODEL_SAVE_PATH)
                print(f"Saved new best model to {MODEL_SAVE_PATH} with accuracy: {best_acc:.4f}")

    # Training time calculate karo
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:4f}')

    # Best model weights load karo (agar validation hua tha)
    if val_loader is not None and best_acc > 0: # Ensure validation happened and accuracy improved
        model.load_state_dict(best_model_wts)
    elif val_loader is None: # Agar validation nahi hua, toh last epoch wala model save karo
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"No validation performed. Saved model from last epoch to {MODEL_SAVE_PATH}")
    else: # Validation hua par accuracy improve nahi hui (best_acc = 0), last model save kardo
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Validation accuracy did not improve. Saved model from last epoch to {MODEL_SAVE_PATH}")


    print(f"Model training finished. Final model saved to {MODEL_SAVE_PATH}")


# --- Script Execution ---
if __name__ == '__main__':
    # Ensure data directory exists before starting
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        print("Please create it and add 'train'/'val' subdirectories with image classes.")
    else:
        train_model() # Training function call karo