import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os

# CONFIG
IMG_SIZE = 96
SEQUENCE_LENGTH = 20

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"D:\DataScience\Deep Learning Project\Video_Classifier\best_model.h5")

model = load_model()

# FRAME EXTRACTION
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

# PREDICTION
def predict_video(video_path):
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)

    prediction = model.predict(frames)[0][0]

    if prediction > 0.5:
        return "Violence", prediction
    else:
        return "Non-Violence", 1 - prediction

# UI
st.title("Violence Detection System")
st.write("Upload a video to detect whether it contains violence.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)

    if st.button("Predict"):
        with st.spinner("Analyzing video..."):
            label, confidence = predict_video(tfile.name)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.2f}")

st.markdown("""
---
<div style='text-align: center; color: gray; font-size: 14px;'>
    Created by <b>Soumya Choudhury</b>
</div>
""", unsafe_allow_html=True)