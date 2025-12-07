import os
from PIL import Image

# Folders ke naam define karo
base_dir = "data"
modes = ["train", "val"]
classes = ["human", "no_human"]


def create_dummy_data():
    for mode in modes:
        for class_name in classes:
            # Folder ka path banao (e.g., data/train/human)
            path = os.path.join(base_dir, mode, class_name)

            # Agar folder nahi hai toh banao
            os.makedirs(path, exist_ok=True)
            print(f"Checking folder: {path}")

            # Har folder mein 5 nakli images banao
            for i in range(5):
                img_path = os.path.join(path, f"dummy_{i}.jpg")

                # Agar image pehle se nahi hai tabhi banao
                if not os.path.exists(img_path):
                    # Alag color use karte hain taaki thoda farak rahe
                    color = 'red' if class_name == 'human' else 'blue'
                    img = Image.new('RGB', (224, 224), color=color)
                    img.save(img_path)
                    print(f"Created: {img_path}")
                else:
                    print(f"Already exists: {img_path}")


if __name__ == "__main__":
    create_dummy_data()
    print("\n✅ Dataset ban gaya! Ab 'python train.py' run karo.")