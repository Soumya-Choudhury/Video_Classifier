import os
import cv2
import numpy as np
from tqdm import tqdm
import gc

# CONFIG
IMG_SIZE = 96
SEQ_LENGTH = 20   
DATA_DIR = "D:/DataScience/Deep Learning Project/Video_Classifier/data"
OUTPUT_DIR = "D:/DataScience/Deep Learning Project/Video_Classifier/data/train/processed"

CATEGORIES = ["NonViolence", "Violence"]

# Create output folders
for split in ["train", "val"]:
    for category in CATEGORIES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)


# FUNCTION: Extract Frames
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < SEQ_LENGTH:
        cap.release()
        return None

    # Uniform sampling
    skip = total_frames // SEQ_LENGTH

    for i in range(SEQ_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        success, frame = cap.read()

        if not success:
            cap.release()
            return None

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

    cap.release()

    # Convert to float32
    frames = np.array(frames, dtype=np.float32) / 255.0

    return frames


# FUNCTION: Process Folder
def process_split(split):
    print(f"\n Processing {split.upper()} data...")

    for category in CATEGORIES:
        input_path = os.path.join(DATA_DIR, split, category)
        output_path = os.path.join(OUTPUT_DIR, split, category)

        files = [f for f in os.listdir(input_path) if f.endswith(('.mp4', '.avi', '.mov'))]

        for file in tqdm(files, desc=f"{split}-{category}"):
            video_path = os.path.join(input_path, file)

            try:
                frames = extract_frames(video_path)

                if frames is None:
                    continue

                # Save as .npy file
                save_path = os.path.join(output_path, file.split('.')[0] + ".npy")
                np.save(save_path, frames)

                # Free memory
                del frames
                gc.collect()

            except Exception as e:
                print(f"Skipping {file}: {e}")
                continue


# MAIN
if __name__ == "__main__":
    print(" Starting preprocessing...")

    process_split("train")
    process_split("val")

    print("\n Preprocessing completed successfully!")