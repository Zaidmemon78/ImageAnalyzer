import os
import shutil

# Classes ke naam
classes = ['human', 'no_human']


def move_images():
    print("Images shift kar rahe hain...")

    for class_name in classes:
        # Paths define karo
        train_path = os.path.join("data", "train", class_name)
        val_path = os.path.join("data", "val", class_name)

        # Validation folder banao agar nahi hai
        os.makedirs(val_path, exist_ok=True)

        # Train folder ki saari images ki list lo
        images = [f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        total_images = len(images)
        print(f"Checking {class_name}: Total {total_images} images hain.")

        if total_images == 0:
            print(f"⚠️ Warning: {class_name} folder khaali hai!")
            continue

        # 20% images ko validation mein bhejenge
        move_count = int(total_images * 0.2)

        # Agar images bohot kam hain, toh kam se kam 2 toh bhejo
        if move_count < 2 and total_images > 2:
            move_count = 2

        print(f"Moving {move_count} images to Validation folder...")

        # Move karna shuru karo
        for i in range(move_count):
            src = os.path.join(train_path, images[i])
            dst = os.path.join(val_path, images[i])

            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"Error moving file: {e}")


if __name__ == "__main__":
    move_images()
    print("\n✅ Transfer Complete!")
    print("👉 Ab 'python train.py' run karo.")