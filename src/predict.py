import cv2
import numpy as np
import tensorflow as tf

# CONFIG
IMG_SIZE = 96
SEQUENCE_LENGTH = 20

# LOAD MODEL
model = tf.keras.models.load_model("best_model.h5")

# FRAME EXTRACTION
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, SEQUENCE_LENGTH).astype(int)

    current_frame = 0
    target_idx = 0

    while cap.isOpened() and target_idx < len(frame_indices):
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame == frame_indices[target_idx]:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frames.append(frame)
            target_idx += 1

        current_frame += 1

    cap.release()

    while len(frames) < SEQUENCE_LENGTH:
        frames.append(frames[-1])

    return np.array(frames)

# PREDICTION
def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)  # shape: (1, 20, 96, 96, 3)

    prediction = model.predict(frames)[0][0]

    if prediction > 0.5:
        label = "Violence"
    else:
        label = "Non-Violence"

    return label, prediction

# TEST
if __name__ == "__main__":
    video_path = r"D:\DataScience\Deep Learning Project\Video_Classifier\data\test data\Violence\V_421.mp4"  

    label, confidence = predict_video(video_path)

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.4f}")