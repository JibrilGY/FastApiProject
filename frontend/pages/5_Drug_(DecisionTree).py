import requests
import streamlit as st

st.title("💊 Drug200 Classification System")
st.markdown("Enter patient features to predict the most suitable drug type.")
st.divider()

col1, col2 = st.columns(2)
with col1:
  age = st.number_input("Age", 15, 80, 30)
  sex = st.selectbox(
      "Sex",
      ["M", "F"],
      format_func=lambda x: "Male" if x == "M" else "Female",
  )
  bp = st.selectbox("Blood Pressure (BP)", ["HIGH", "LOW", "NORMAL"])

with col2:
  cholesterol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
  na_to_k = st.number_input(
      "Sodium to Potassium Ratio (Na_to_K)", 5.0, 40.0, 15.0, 0.1
  )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Drug 🔍", use_container_width=True, type="primary"):
  payload = {
      "Age": age,
      "Sex": sex,
      "BP": bp,
      "Cholesterol": cholesterol,
      "Na_to_K": na_to_k,
  }

  try:
    response = requests.post("http://127.0.0.1:8000/drug/predict", json=payload)
    if response.status_code == 200:
      result = response.json()
      prediction = result["prediction"]
      st.divider()
      st.success(f"✅ **Recommended Drug / Class:** {prediction}")
    else:
      st.error(f"API Error: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error(
        "⚠️ **Connection Error:** FastAPI server is not running or unreachable."
    )