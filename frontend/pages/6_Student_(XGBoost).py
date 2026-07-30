import requests
import streamlit as st

st.title("🎓 Student Placement Prediction System")
st.markdown("Enter student details to predict academic placement status.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    study_hours = st.number_input(
        "Daily Study Hours", 0.0, 24.0, 4.0, 0.5
    )
    attendance = st.number_input(
        "Attendance Rate (%)", 0.0, 100.0, 85.0, 1.0
    )
    sleep_hours = st.number_input(
        "Daily Sleep Hours", 0.0, 24.0, 7.0, 0.5
    )

with col2:
    assignments_completed = st.number_input(
        "Completed Assignments", 0.0, 100.0, 10.0, 1.0
    )
    previous_score = st.number_input(
        "Previous Score / GPA", 0.0, 100.0, 75.0, 1.0
    )

st.markdown("<br>", unsafe_allow_html=True)
if st.button(
        "Predict Placement Status 🔍", use_container_width=True, type="primary"
):
    payload = {
        "study_hours": study_hours,
        "assignments_completed": assignments_completed,
        "previous_score": previous_score,
        "attendance": attendance,
        "sleep_hours": sleep_hours,
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/student/predict", json=payload
        )
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probabilities = result["probability"]

            st.divider()
            if prediction == 1:
                st.success(f"✅ **Result:** Placed (Probability: {probabilities['placed'] * 100:.2f}%)")
            else:
                st.error(f"❌ **Result:** Not Placed (Probability: {probabilities['not_placed'] * 100:.2f}%)")
        else:
            st.error(f"API Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Connection Error:** FastAPI server is not running.")