import os
import cv2
import numpy as np

IMG_SIZE = 96
SEQUENCE_LENGTH = 20

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // SEQUENCE_LENGTH, 1)

    for i in range(SEQUENCE_LENGTH):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.array(frames)

# PATHS
INPUT_DIR = r"data/test data"
OUTPUT_DIR = r"data/processed/test"

for category in ["Violence", "NonViolence"]:
    input_path = os.path.join(INPUT_DIR, category)
    output_path = os.path.join(OUTPUT_DIR, category)

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_path, file)

            frames = extract_frames(video_path)

            save_path = os.path.join(output_path, file.replace(".mp4", ".npy"))
            np.save(save_path, frames)

            print(f"Processed: {file}")