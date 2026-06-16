import streamlit as st
import pickle
import numpy as np

# Load trained model and scaler
model = pickle.load(open("burnout_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("Student Burnout Prediction System")

st.write("Enter student information to predict burnout level.")

# User inputs
daily_sleep_hours = st.number_input(
    "Daily Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

screen_time_hours = st.number_input(
    "Screen Time Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

physical_activity_hours = st.number_input(
    "Physical Activity Hours",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ["Average", "Good", "Poor"]
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

# Encoding for sleep quality
sleep_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}

# Prediction button
if st.button("Predict Burnout Level"):

    input_data = np.array([[
        daily_sleep_hours,
        screen_time_hours,
        physical_activity_hours,
        sleep_map[sleep_quality],
        cgpa
    ]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    # Decode burnout level
    burnout_map = {
        0: "High",
        1: "Low",
        2: "Medium"
    }

    st.success(
        f"Predicted Burnout Level: {burnout_map[prediction[0]]}"
    )