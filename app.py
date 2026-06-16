import streamlit as st
import pickle
import numpy as np



model = pickle.load(open("burnout_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.set_page_config(
    page_title="Student Burnout Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Burnout Prediction System")

st.markdown("""
This application predicts a student's burnout level based on lifestyle and academic factors using a Support Vector Machine (SVM) model.
""")

st.divider()


daily_sleep_hours = st.slider(
    "🛌 Daily Sleep Hours",
    min_value=0,
    max_value=24,
    value=7
)

screen_time_hours = st.slider(
    "📱 Screen Time Hours",
    min_value=0,
    max_value=24,
    value=5
)

physical_activity_hours = st.slider(
    "🏃 Physical Activity Hours",
    min_value=0,
    max_value=10,
    value=1
)

sleep_quality = st.selectbox(
    "😴 Sleep Quality",
    ["Average", "Good", "Poor"]
)

cgpa = st.number_input(
    "📚 CGPA",
    min_value=0.0,
    max_value=4.0,
    value=3.0,
    step=0.01
)




sleep_map = {
    "Average": 0,
    "Good": 1,
    "Poor": 2
}

# ==========================
# Prediction
# ==========================

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

    burnout_map = {
        0: "High",
        1: "Low",
        2: "Medium"
    }

    result = burnout_map[prediction[0]]

    st.success(f"Predicted Burnout Level: {result}")

   
