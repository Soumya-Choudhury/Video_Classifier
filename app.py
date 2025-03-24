import streamlit as st
import numpy as np
from catboost import CatBoostClassifier

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Mapping from numerical predictions to obesity categories
obesity_categories = {
    0: "Insufficient Weight",
    1: "Normal Weight",
    2: "Overweight Level I",
    3: "Overweight Level II",
    4: "Obesity Type I",
    5: "Obesity Type II",
    6: "Obesity Type III"
}

# Streamlit UI
st.title("Obesity Category Predictor")
st.write("Enter your details below to predict your obesity category.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=5, max_value=100, value=25)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0)
fcvc = st.number_input("Frequency of Consumption of Vegetables (FCVC) [0-3]", min_value=1.0, max_value=3.0)
ch2o = st.number_input("Daily Water Intake (CH2O) [0-3]", min_value=1.0, max_value=3.0)
faf = st.number_input("Physical Activity Frequency (FAF) [0-3]", min_value=1.0,max_value=3.0)

# Convert gender to numerical (assuming 0 for Female, 1 for Male)
gender_num = 1 if gender == "Male" else 0

# Prepare input data
input_data = np.array([[gender_num, age, weight, fcvc, ch2o, faf]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # Get the predicted class
    category = obesity_categories.get(int(prediction), "Unknown Category")
    
    # Display the result
    st.success(f"Predicted Obesity Category: **{category}**")
