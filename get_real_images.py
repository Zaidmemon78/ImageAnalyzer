import os
import shutil
from icrawler.builtin import BingImageCrawler
from PIL import Image

# --- CONFIGURATION ---
TOTAL_IMAGES = 60  # Total images per class
TRAIN_COUNT = 50  # Isme se 50 Training mein jayengi
VAL_COUNT = 10  # Aur 10 Validation mein


def create_folders():
    """Folders banata hai aur purana data saaf karta hai"""
    paths = [
        "data/train/human", "data/train/no_human",
        "data/val/human", "data/val/no_human"
    ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)  # Purana folder delete
        os.makedirs(path)  # Naya folder banao


def clean_images(folder_path):
    """Corrupted images ko delete karta hai jo open nahi ho rahin"""
    print(f"Checking images in {folder_path}...")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Check karo image sahi hai ya nahi
        except (IOError, SyntaxError):
            print(f"Bad file deleted: {filename}")
            os.remove(file_path)


def split_data(class_name, temp_folder):
    """Images ko Train aur Val folders mein divide karta hai"""
    images = [f for f in os.listdir(temp_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Train folder mein move karo
    train_dest = f"data/train/{class_name}"
    for i in range(min(len(images), TRAIN_COUNT)):
        shutil.move(os.path.join(temp_folder, images[i]), os.path.join(train_dest, images[i]))

    # Val folder mein move karo
    val_dest = f"data/val/{class_name}"
    remaining = [f for f in os.listdir(temp_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    for i in range(min(len(remaining), VAL_COUNT)):
        shutil.move(os.path.join(temp_folder, remaining[i]), os.path.join(val_dest, remaining[i]))


def download_class(keyword, class_name):
    temp_dir = f"temp_{class_name}"

    # Download start
    print(f"\nDownloading '{keyword}'...")
    crawler = BingImageCrawler(storage={'root_dir': temp_dir})
    crawler.crawl(keyword=keyword, max_num=TOTAL_IMAGES, filters=None, file_idx_offset=0)

    # Bad images delete karo
    clean_images(temp_dir)

    # Sahi folders mein daalo
    split_data(class_name, temp_dir)

    # Temp folder uda do
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    create_folders()

    # 1. Download Humans
    download_class("full body person standing street photo", "human")

    # 2. Download No Humans (Empty rooms/Nature)
    download_class("empty living room interior photo", "no_human")

    print("\n✅ Sab set hai! Ab 'python train.py' run karo.")