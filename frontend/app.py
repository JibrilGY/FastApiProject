import requests
import streamlit as st

st.set_page_config(
    page_title="Multi-Model ML Platformu",
    page_icon="🤖",
    layout="centered",
)

st.title("🚀 Makine Öğrenmesi Tahmin Platformu")
st.markdown(
    "Hoş geldiniz! Bu platform, FastAPI tabanlı arka uç servisleri ve"
    " Scikit-Learn Pipeline yapıları üzerine kurulmuştur."
)
st.info(
    "👉 Modeller arasında geçiş yapmak için sol menüyü (sidebar) kullanabilirsiniz."
)

st.markdown("### 📊 Sistemde Yer Alan Modeller:")

# Backend bağlantı kontrolü ve model listesi
try:
  response = requests.get("http://127.0.0.1:8000/models", timeout=2)
  if response.status_code == 200:
    raw_list = response.json().get("models", [])
    model_list = [m for m in raw_list if "model_columns" not in m]
    st.success(f"✅ Backend bağlantısı aktif! Aktif Model Sayısı: {len(model_list)}")
  else:
    st.warning("⚠️ Backend'den model listesi alınamadı.")
except Exception:
  st.error(
      "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor."
      " (`uvicorn main:app --reload` komutuyla başlattığından emin ol)"
  )

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
  st.markdown("#### 🚢 Titanic")
  st.write("Yolcu hayatta kalma analizi.")
  st.markdown("#### 🎗️ Kanser (SVM)")
  st.write("SVM tabanlı tümör teşhisi.")
with col2:
  st.markdown("#### 💳 Banka Kredisi")
  st.write("Kredi onay risk analizi.")
  st.markdown("#### 🎗️ Kanser (KNN)")
  st.write("KNN tabanlı tümör teşhisi.")
with col3:
  st.markdown("#### 💊 Drug200")
  st.write("Hasta ilaç sınıflandırması.")
  st.markdown("#### 🎓 Öğrenci Durumu")
  st.write("Öğrenci başarı analizi.")