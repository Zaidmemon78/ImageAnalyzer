# 🕵️ Human Detector AI

A Deep Learning project that detects the presence of humans in images and live video feeds using **Transfer Learning (MobileNetV2)**. 

This model is built using **PyTorch** and can classify inputs into two categories: `Human` and `No Human`. It achieves approximately **92% accuracy**.

## 🚀 Features
- **Automated Dataset Creation:** Scripts to download real-world images from the web automatically.
- **Transfer Learning:** Uses a pre-trained MobileNetV2 model for faster and accurate training.
- **Real-Time Detection:** Live detection using a webcam with OpenCV.
- **Image Analysis:** Predicts classes for individual image files.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** PyTorch, Torchvision, OpenCV, PIL, icrawler
- **Model Architecture:** MobileNetV2

## 📂 Project Structure
image_AI/
├── data/ # Stores training and validation images
├── models/ # Stores the saved trained model (.pth)
├── get_real_images.py # Script to download dataset from Bing
├── train.py # Script to train the model
├── predict.py # Script to test on a single image
├── realtime_detect.py # Script for live webcam detection
├── model.py # Model architecture definition
└── data_loader.py # Data loading and transformation logic
code
Code
## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
   cd image_AI
Install dependencies:
code
Bash
pip install torch torchvision opencv-python icrawler pillow
🏃‍♂️ How to Run
1. Generate Dataset
If you don't have images, run this script to download them automatically:
code
Bash
python get_real_images.py
2. Train the Model
Train the AI on the downloaded images:
code
Bash
python train.py
This will save the model as human_detector_mobilenetv2.pth inside the models/ folder.
3. Live Webcam Detection (Coolest Feature! 🎥)
To detect humans in real-time using your webcam:
code
Bash
python realtime_detect.py
Press q to exit the camera window.
4. Predict on a Single Image
To test a specific image file:
code
Bash
python predict.py --image "path/to/your/image.jpg"
📊 Results
Training Accuracy: ~100%
Validation Accuracy: ~92%
Status: The model successfully distinguishes between humans and empty rooms/objects.
