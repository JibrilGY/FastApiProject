import requests
import streamlit as st

st.title("🎗️ Meme Kanseri Teşhis Sistemi (SVM)")
st.markdown(
    "Tümör ölçüm değerlerini girerek iyi huylu veya kötü huylu olma"
    " durumunu analiz edin."
)
st.divider()

features_list = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

defaults = {
    "radius_mean": 14.12,
    "texture_mean": 19.29,
    "perimeter_mean": 91.97,
    "area_mean": 654.89,
    "smoothness_mean": 0.096,
    "compactness_mean": 0.104,
    "concavity_mean": 0.089,
    "concave points_mean": 0.049,
    "symmetry_mean": 0.181,
    "fractal_dimension_mean": 0.063,
    "radius_se": 0.405,
    "texture_se": 1.217,
    "perimeter_se": 2.866,
    "area_se": 40.34,
    "smoothness_se": 0.007,
    "compactness_se": 0.025,
    "concavity_se": 0.032,
    "concave points_se": 0.012,
    "symmetry_se": 0.021,
    "fractal_dimension_se": 0.004,
    "radius_worst": 16.27,
    "texture_worst": 25.68,
    "perimeter_worst": 107.26,
    "area_worst": 880.58,
    "smoothness_worst": 0.132,
    "compactness_worst": 0.254,
    "concavity_worst": 0.272,
    "concave points_worst": 0.115,
    "symmetry_worst": 0.290,
    "fractal_dimension_worst": 0.084,
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
if st.button("Teşhisi Başlat 🔍", use_container_width=True, type="primary"):
  try:
    response = requests.post("http://127.0.0.1:8000/predict/cancer_svm", json=payload)
    if response.status_code == 200:
      result = response.json()
      prediction = result["prediction"]
      prob_benign = result["probability_benign"]
      prob_malignant = result["probability_malignant"]
      st.divider()
      if prediction == 0:
        st.success(
            f"✅ **Sonuç: İyi Huylu (Benign)** (İyi Huylu Olasılığı:"
            f" %{prob_benign * 100:.2f})"
        )
      else:
        st.error(
            f"⚠️ **Sonuç: Kötü Huylu (Malignant)** (Kötü Huylu Olasılığı:"
            f" %{prob_malignant * 100:.2f})"
        )
      st.progress(float(prob_malignant))
    else:
      st.error(f"API Hatası: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error("⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor.")