import requests
import streamlit as st

st.set_page_config(
    page_title="Multi-Model ML Platform",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🚀 Machine Learning Prediction Platform")
st.markdown(
    "Welcome! This platform is built on FastAPI-based backend services "
    "and Scikit-Learn Pipeline architectures."
)
st.info(
    "👉 Use the sidebar to switch between different models."
)

st.markdown("### 📊 Available Models in the System:")

# Backend connection check and model list
try:
    response = requests.get("http://127.0.0.1:8000/models", timeout=2)
    if response.status_code == 200:
        raw_list = response.json().get("models", [])
        model_list = [m for m in raw_list if "model_columns" not in m]
        st.success(f"✅ Backend connection active! Active Models Count: {len(model_list)}")
    else:
        st.warning("⚠️ Could not retrieve model list from backend.")
except Exception:
    st.error(
        "⚠️ **Connection Error:** FastAPI server is not running. "
        "(Make sure you started it with `uvicorn main:app --reload`)"
    )

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### 🚢 Titanic")
    st.write("Passenger survival analysis.")
    st.markdown("#### 🎗️ Cancer (SVM)")
    st.write("SVM-based tumor diagnosis.")
with col2:
    st.markdown("#### 💳 Bank Loan")
    st.write("Loan approval risk analysis.")
    st.markdown("#### 🎗️ Cancer (KNN)")
    st.write("KNN-based tumor diagnosis.")
with col3:
    st.markdown("#### 💊 Drug200")
    st.write("Patient drug classification.")
    st.markdown("#### 🎓 Student Status")
    st.write("Student performance analysis.")