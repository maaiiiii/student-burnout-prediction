import streamlit as st
import pickle
import numpy as np

# ==========================
# Load Model and Scaler
# ==========================

model = pickle.load(open("burnout_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Student Burnout Prediction",
    page_icon="🎓",
    layout="centered"
)

# ==========================
# Title
# ==========================

st.title("🎓 Let's Predict Your Burnout")

st.markdown("""
This application predicts a student's burnout level based on lifestyle and academic factors.
made by umi maisarah CB23036
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



if st.button("Predict Burnout Level"):

    input_data = np.array([[
        daily_sleep_hours,
        screen_time_hours,
        physical_activity_hours,
        sleep_map[sleep_quality],
        cgpa
    ]])


    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    burnout_map = {
        0: "High",
        1: "Low",
        2: "Medium"
    }

    result = burnout_map[prediction[0]]

    st.success(f"Predicted Burnout Level: {result}")

    # Additional message
    if result == "High":
        st.error("⚠️ Please get some rest")
    elif result == "Medium":
        st.warning("⚠️ Please take care of yourself")
    else:
        st.info("✅ good!.")
