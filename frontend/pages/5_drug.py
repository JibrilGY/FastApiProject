import requests
import streamlit as st

st.title("💊 Drug200 İlaç Sınıflandırma Sistemi")
st.markdown("Hasta özelliklerini girerek en uygun ilaç türünü tahmin edin.")
st.divider()

col1, col2 = st.columns(2)
with col1:
  age = st.number_input("Yaş (Age)", 15, 80, 30)
  sex = st.selectbox(
      "Cinsiyet (Sex)",
      ["M", "F"],
      format_func=lambda x: "Erkek" if x == "M" else "Kadın",
  )
  bp = st.selectbox("Kan Basıncı (BP)", ["HIGH", "LOW", "NORMAL"])

with col2:
  cholesterol = st.selectbox("Kolesterol (Cholesterol)", ["HIGH", "NORMAL"])
  na_to_k = st.number_input(
      "Sodyum / Potasyum Oranı (Na_to_K)", 5.0, 40.0, 15.0, 0.1
  )

st.markdown("<br>", unsafe_allow_html=True)
if st.button("İlacı Tahmin Et 🔍", use_container_width=True, type="primary"):
  payload = {
      "Age": age,
      "Sex": sex,
      "BP": bp,
      "Cholesterol": cholesterol,
      "Na_to_K": na_to_k,
  }

  try:
    # URL güncellendi: /drug/predict
    response = requests.post("http://127.0.0.1:8000/drug/predict", json=payload)
    if response.status_code == 200:
      result = response.json()
      prediction = result["prediction"]
      st.divider()
      st.success(f"✅ **Önerilen İlaç / Sınıf Sonucu:** {prediction}")
    else:
      st.error(f"API Hatası: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error("⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor.")