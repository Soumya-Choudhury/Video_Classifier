import os
import shutil
import random

random.seed(42)

# Paths
SOURCE_DIR = "data/Real Life Violence Dataset"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

# Split ratio
SPLIT_RATIO = 0.85  # 85% train, 15% validation


def split_category(category):
    source_path = os.path.join(SOURCE_DIR, category)
    train_path = os.path.join(TRAIN_DIR, category)
    val_path = os.path.join(VAL_DIR, category)

    # Create directories if not exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Get all video files
    files = [f for f in os.listdir(source_path) if f.endswith(('.mp4', '.avi', '.mov'))]

    print(f"\nProcessing {category}...")
    print(f"Total files found: {len(files)}")

    # Shuffle files
    random.shuffle(files)

    # Split
    split_index = int(len(files) * SPLIT_RATIO)

    train_files = files[:split_index]
    val_files = files[split_index:]

    # Copy files
    for f in train_files:
        shutil.copy2(
            os.path.join(source_path, f),
            os.path.join(train_path, f)
        )

    for f in val_files:
        shutil.copy2(
            os.path.join(source_path, f),
            os.path.join(val_path, f)
        )

    print(f"{category} → Train: {len(train_files)}, Val: {len(val_files)}")


if __name__ == "__main__":
    print("Starting dataset split...")

    categories = ["Violence", "NonViolence"]

    for category in categories:
        split_category(category)

    print("\n Data splitting completed successfully!")