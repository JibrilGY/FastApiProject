import requests
import streamlit as st

st.title("🎗️ Breast Cancer Prediction System (SVM)")
st.markdown(
    "Analyze tumor measurement values to determine benign or malignant status using the SVM algorithm."
)
st.divider()

# ANOVA - 10 Selections
features_list = [
    "radius_mean",
    "texture_mean",
    "smoothness_mean",
    "compactness_mean",
    "radius_se",
    "compactness_se",
    "concave points_se",
    "smoothness_worst",
    "symmetry_worst",
    "radius_worst",
]

defaults = {
    "radius_mean": 14.12,
    "texture_mean": 19.29,
    "smoothness_mean": 0.096,
    "compactness_mean": 0.104,
    "radius_se": 0.405,
    "compactness_se": 0.025,
    "concave points_se": 0.012,
    "smoothness_worst": 0.132,
    "symmetry_worst": 0.290,
    "radius_worst": 16.27,
}

payload = {}
col1, col2 = st.columns(2)

for i, col_name in enumerate(features_list):
  target_col = col1 if i % 2 == 0 else col2
  with target_col:
    payload[col_name] = st.number_input(
        f"{col_name}",
        value=float(defaults.get(col_name, 0.0)),
        format="%.4f",
        key=f"cancer_svm_{col_name}",
    )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Start 🔍", use_container_width=True, type="primary"):
  try:
    response = requests.post("http://127.0.0.1:8000/cancer-svm/predict-svm", json=payload)

    if response.status_code == 200:
      result = response.json()
      prediction = result["prediction"]

      prob_benign = result["probability"]["benign"]
      prob_malignant = result["probability"]["malignant"]

      st.divider()
      if prediction == 0:
        st.success(
            f"✅ **Result: Benign ** (Good:"
            f" %{prob_benign * 100:.2f})"
        )
      else:
        st.error(
            f"⚠️ **Result: Malignant ** (Bad:"
            f" %{prob_malignant * 100:.2f})"
        )
      st.progress(float(prob_malignant))
    else:
      st.error(f"API Error: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error("⚠️ **Connection Error:** FastAPI server is not working.")