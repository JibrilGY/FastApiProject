import requests
import streamlit as st

st.title("🎓 Öğrenci Durum ve Başarı Tahmin Sistemi")
st.markdown("Öğrenci bilgilerini girerek akademik başarı durumunu tahmin edin.")
st.divider()

col1, col2 = st.columns(2)
with col1:
  study_hours = st.number_input(
      "Günlük Çalışma Saati (Study Hours)", 0.0, 24.0, 4.0, 0.5
  )
  attendance = st.number_input(
      "Devam Oranı / Yüzdesi (Attendance)", 0.0, 100.0, 85.0, 1.0
  )
  sleep_hours = st.number_input("Uyku Saati (Sleep Hours)", 0.0, 24.0, 7.0, 0.5)
  internet_usage = st.number_input(
      "İnternet Kullanımı (Saat/Gün)", 0.0, 24.0, 2.0, 0.5
  )

with col2:
  assignments_completed = st.number_input(
      "Tamamlanan Ödev Sayısı", 0.0, 100.0, 10.0, 1.0
  )
  previous_score = st.number_input(
      "Önceki Not / Puan (Previous Score)", 0.0, 100.0, 75.0, 1.0
  )
  exam_score = st.number_input("Sınav Puanı (Exam Score)", 0.0, 100.0, 70.0, 1.0)

st.markdown("<br>", unsafe_allow_html=True)
if st.button(
    "Başarı Durumunu Tahmin Et 🔍", use_container_width=True, type="primary"
):
  payload = {
      "study_hours": study_hours,
      "attendance": attendance,
      "sleep_hours": sleep_hours,
      "internet_usage": internet_usage,
      "assignments_completed": assignments_completed,
      "previous_score": previous_score,
      "exam_score": exam_score,
  }

  try:
    response = requests.post(
        "http://127.0.0.1:8000/predict/student", json=payload
    )
    if response.status_code == 200:
      result = response.json()
      prediction = result["prediction"]
      st.divider()
      st.success(f"✅ **Sonuç / Tahmin:** {prediction}")
    else:
      st.error(f"API Hatası: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error("⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor.")