## Violence Detection in Videos Using Deep Learning

This project focuses on building a deep learning based video classification system that can automatically detect whether a video contains violent or non-violent content. The goal is to simulate a real-world application like CCTV surveillance or public safety systems where detecting the violence quickly is critical. The project is divided into two major phases:
- **Phase 1:Initial Experimentation**
  - Began with understanding the dataset where there were 1000 videos of each violence and non-violence content.
  - Used CNN + LSTM model for building the model to understand how well video data can be used for violence detection.
  - Analyzed the overall performance of the model against the training and validation data through performance metrics.

- **Phase 2:Final Experimentation**

## Phase 1:Initial Experimentation

As part of the project, I began by building a baseline deep learning model to understand how well video data can be used for violence detection. This initial phase focused on combining spatial and temporal learning using a **CNN + LSTM (CRNN architecture)**.

At the beginning, the problem was framed as a binary classification task:
- Class 0 (Non-Violence)
- Class 1 (Violence)

However, unlike image classification, this is a video problem, which introduces an extra dimension "time". A single frame is not enough and we need to understand motion and sequence of events.

---

## Dataset
- **Dataset Used:** [Real Life Violence Situations Dataset on Kaggle](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)

### Data Exploration

* Visualized sample frames from both **Violence** and **Non-Violence** classes
* Analyzed distribution of video lengths to understand variability across samples
* Selected a fixed sequence length of **20 frames per video** to maintain consistency

---

## Technologies & Tools

- **Languages & Libraries:** Python, OpenCV, NumPy, Matplotlib, TensorFlow/Keras, Scikit-learn
- **Techniques Used:** 
  - Frame sampling and resizing
  - TimeDistributed CNN layers for spatial features
  - LSTM layer for temporal sequence learning
  - Binary classification with performance metrics

---

## Project Structure

- `Video_classifier.ipynb` – Full training pipeline

### Model Architecture

The baseline model was designed using a **Convolutional Recurrent Neural Network (CRNN)**:

* **TimeDistributed CNN layers** for extracting spatial features from each frame
* **MaxPooling layers** for dimensionality reduction
* **LSTM layer** to capture temporal dependencies across frames
* Fully connected (**Dense**) layers for final binary classification

Why CNN and LSTM?
- CNN(Convolutional Neural Network) is specialized for grid like data such as image, video and audio based data. It automatically learns features i.e., edges, shapes and objects from images. Here it helped to extract features from each frame i.e., what is in the image.

- LSTM (Long Short Term Memory) is a special type of RNN (Recurrent Neural Network) used for sequence data. It remembers long term dependencies. Here it helped to understand the sequence of frames i.e., what happens over time.
---

### Training Performance

After training for 10 epochs:

* **Validation Accuracy:** 80%
* **Validation Loss:** 0.64
* **Precision:** 0.87
* **Recall:** 0.70

---

### Classification Report

```
              precision    recall    f1-score   support

Non-Violence     0.75       0.90       0.82       201
Violence         0.87       0.70       0.78       199

accuracy                                0.80       400
macro avg         0.81       0.80       0.80       400
weighted avg      0.81       0.80       0.80       400
```

---

### Key Observations

* The model achieved **good precision**, meaning most predicted violence cases were correct
* However, the **recall for violence detection was relatively low (70%)**, indicating that some violent videos were being missed
* This highlighted a critical issue of **false negatives**, which is especially important in safety-related applications
* The relatively high loss indicated that the model was still struggling to generalize effectively.
---

### Improvements Based on This Analysis

Based on insights from this baseline experiment, several improvements were implemented in the final system:

* Designed a **robust preprocessing pipeline** (frame extraction, resizing, normalization)
* Introduced a **custom data generator** for efficient training
* Improved model generalization and performance

---
## Phase 2:Final Experimentation

## Dataset Preparation
I separated 30 videos of each violence and non-violence category to a test folder and the remaining 970 videos of each category is being used to split between training(75%) and validation set(25%).
We structured the data into: 
```md
data/
├── train/
├── val/
├── test/
```
Instead of feeding raw .mp4 videos directly to the model (which is inefficient and slow), I converted videos into frames.

## Preprocessing Pipeline
- Extracted frames from each video.
- Resized frames to: 96x96
- Selected a fixed number of frames: SEQUENCE_LENGTH = 20
- Normalized pixel values by dividing frames with 255.0
- Saved preprocessed data as .npy files

Why this matters?
- Ensures uniform input shape
- Reduces computational load
- Makes training faster & stable

## Challenges
1) Problem: 
- OpenCV frame seeking is not precise
- Different runs → different frames → inconsistent predictions
  
  Solution:
- Switched to a deterministic frame extraction approach:
  - Sequential Reading
  - Fixed frame indices using np.linspace
This ensured same frames every time and stable predictions.

## Custom Data Generator
Instead of loading all data into memory, a custom VideoGenerator (Keras Sequence) was implemented.

What it does?
- Loads .npy files batch-wise
- Returns: 
  X-> (batch_size,20,96,96,3)
  y-> labels

Why this is important?
- Prevents memory overflow
- Scales to large datasets
- Industry-standard approach

## Model Architecture
The final model is built using a Transfer Learning based CNN-GRU architecture designed to capture both spatial and temporal information from video data.

### Key Components:

- **MobileNetV2 (Pretrained CNN):**
  - Used as a feature extractor (trained on ImageNet)
  - Applied to each frame using a TimeDistributed layer
  - Helps in extracting spatial features such as objects, shapes, and patterns

- **TimeDistributed Layer:**
  - Applies the CNN model independently to each frame in the sequence
  - Converts video input into a sequence of feature vectors

- **GRU (Gated Recurrent Unit):**
  - Captures temporal dependencies across frames
  - Learns motion patterns and sequence behavior in videos
  - GRU was chosen over LSTM because it has fewer parameters, leading to faster training and reduced computational cost while maintaining comparable performance.

- **Dense + Dropout Layers:**
  - Perform final classification
  - Dropout helps reduce overfitting

What the model learns?
- Body movement patterns
- Aggressive actions
- Scene context

### Model Flow
```md
Video Frames (20)
        ↓
MobileNetV2 (per frame)
        ↓
Feature Vectors
        ↓
GRU Layer
        ↓
Dense + Dropout
        ↓
Output (Violence / Non-Violence)
```
## Training Phase
Trained the model on:
- Training set
- Validation set

The model demonstrated strong learning capability during training. In the initial epochs, both training and validation accuracy improved rapidly, indicating effective feature extraction from video sequences. By epochs 6–7, the model achieved peak validation performance (~95%), showing a good balance between learning and generalization.

In later epochs, training accuracy continued to increase (~99%), while validation accuracy plateaued, indicating mild overfitting. This suggests that the model had already learned the most relevant patterns and further training primarily led to memorization of training data.

Overall, the model showed stable convergence, strong generalization, and high performance on unseen data.

Finally, two models were saved:- 
- best_model.h5: This is the model saved when validation performance was the best. This model performs best on unseen data and avoids overfitting. 

- final_model.h5: This is the model saved at the end of training (Epoch 10). This model may overfit training data and may perform slightly worse on new videos.

EarlyStopping and ModelCheckpoint were used to prevent overfitting and ensure that the best-performing model on validation data was saved

### ✅ Final Outcome

The same preprocessing pipeline used for the training and validation data was applied to the test .mp4 videos. These videos were converted into .npy format to maintain consistency and perform testing on best_model.h5. 

After applying these improvements, the final model achieved:

* **Test Accuracy:** ~91%
* **Improved Recall for Violence Detection:** ~90%
* **Improved Precision for Violence Detection:** ~90%

### Confusion Matrix
[[26  2]
 [ 3 27]]

- True Positives (Violence correctly detected): 27  
- True Negatives (Non-Violence correctly detected): 26  
- False Positives: 2  
- False Negatives: 3  

The low number of false negatives (3) is especially critical, as missing violent events can have serious real-world consequences in surveillance systems.

This progression demonstrates a structured approach of **experiment → analyze → improve → optimize**, which is essential in real-world machine learning workflows.

## Streamlit Application

An interactive web app was built using Streamlit to demonstrate real-time predictions.

### Features:
- Upload video 
- Preview uploaded video
- Predict violence/non-violence
- Display confidence score

### How to Run:

```bash
streamlit run app.py
```

## Final System Pipeline

Video (.mp4)
   ↓
Frame Extraction
   ↓
Preprocessing (resize, normalize)
   ↓
Sequence Formation (20 frames)
   ↓
Feature Extraction (MobileNetV2 - per frame)
   ↓
Temporal Modeling (GRU)
   ↓
Prediction (Sigmoid Output)
   ↓
Final Output (Label + Confidence)
   ↓
Streamlit UI

```md
## Project Structure


Video_Classifier/
│
├── data/
│   ├── Real Life Violence Dataset
│   ├── train/
│   ├── val/
│   ├── processed/test
│   └── test data/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── model.py
│   ├── evaluate.ipynb
│   ├── preprocess_test.py
│   ├── preprocessing.py
│   ├── data_generator.py
│   ├── app.py
│   ├── split_data.py
│   ├── Video_classifier.py
│
├── venv
├── best_model.h5
├── final_model.h5
│
├── .gitignore
├── requirements.txt
└── README.md
```
## Conclusion

This project demonstrates a complete end-to-end machine learning pipeline, from data preprocessing and model building to evaluation and deployment. It highlights the importance of iterative improvement, proper validation, and real-world system design in deep learning applications.

## Key Learnings

- Handling video data requires both spatial and temporal modeling
- Deterministic preprocessing is critical for reproducibility
- Validation metrics are more important than training accuracy
- Overfitting can occur even with high accuracy
- Efficient data pipelines (generators) are essential for scalability

## Future Improvements

- Use 3D CNNs (C3D / I3D) for better spatio-temporal learning
- Integrate real-time webcam inference
- Deploy using Docker + Cloud (AWS/GCP)
- Improve dataset diversity for better generalization

## 🎥 Demo Video

👉 **[Watch Demo Video](https://drive.google.com/file/d/1MZANeKpl3i1E7RMZstinAAo4eIfbuqH3/view?usp=drive_link)**

Or preview below:

<iframe src="https://drive.google.com/file/d/1MZANeKpl3i1E7RMZstinAAo4eIfbuqH3/view?usp=drive_link" 
width="700" height="400"></iframe>

## Acknowledgments

- Dataset by [Mohamed Hany on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/real-life-violence-situations-dataset)
