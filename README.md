
# 🧠 Video Classifier: Violence Detection using CRNN

This project implements a deep learning–based video classification system to detect violence in real-time videos. It uses a hybrid **Convolutional + Recurrent Neural Network (CRNN)** architecture to capture both spatial and temporal features across video frames.

---

## 📂 Dataset

- **Dataset Used:** [Real Life Violence Situations Dataset on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/real-life-violence-situations-dataset)
- **Categories:** 
  - Violence (Label: 1)
  - NonViolence (Label: 0)
- **Preprocessing:** 
  - Extracted and resized 20 frames per video (64×64 pixels)
  - Normalized pixel values for model input

---

## 🔧 Technologies & Tools

- **Languages & Libraries:** Python, OpenCV, NumPy, Matplotlib, TensorFlow/Keras, Scikit-learn
- **Techniques Used:** 
  - Frame sampling and resizing
  - TimeDistributed CNN layers for spatial features
  - LSTM layer for temporal sequence learning
  - Binary classification with performance metrics

---

## 📈 Performance Summary

| Metric      | Score    |
|-------------|----------|
| Accuracy    | 80%      |
| Precision   | 87.4%    |
| Recall      | 69.8%    |
| F1 Score    | 80%      |

The model effectively minimizes false positives, making it suitable for real-world safety and surveillance applications.

---

## 📊 Visualizations

- Sample frame displays from both classes
- Distribution of video lengths across the dataset
- Classification report and model evaluation plots

---

## 🚀 Potential Applications

- Violence detection in public surveillance
- Smart security camera systems
- Automated content moderation in video platforms

---

## 📌 Project Structure

- `video_classifier.py` – Full training pipeline
- `utils/` – Frame extraction and preprocessing scripts
- `visuals/` – Plots for data analysis and evaluation
- `models/` – Saved model weights (optional)

---

## 📬 Contact

Created by **Soumya Choudhury**  
📧 [soumyaslg27@gmail.com](mailto:soumyaslg27@gmail.com)  
🔗 [GitHub](https://github.com/Soumya-Choudhury)  
🔗 [LinkedIn](https://www.linkedin.com/in/soumya-choudhury27)

---

## ⭐ Acknowledgments

- Dataset by [Mohamed Hany on Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/real-life-violence-situations-dataset)
- TensorFlow/Keras for model development
- OpenCV for video frame processing
