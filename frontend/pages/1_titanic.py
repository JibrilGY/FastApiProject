import requests
import streamlit as st

st.title("🚢 Titanic Hayatta Kalma Tahmin Sistemi")
st.markdown(
    "Yolcu bilgilerini girerek hayatta kalma ihtimalini anlık olarak hesaplayın."
)
st.divider()

col1, col2 = st.columns(2)
with col1:
  pclass = st.selectbox(
      "Bilet Sınıfı (Pclass)", [1, 2, 3], format_func=lambda x: f"{x}. Sınıf"
  )
  sex = st.selectbox(
      "Cinsiyet",
      ["male", "female"],
      format_func=lambda x: "Erkek" if x == "male" else "Kadın",
  )
  title_options = (
      ["Mr", "Master", "Rare"] if sex == "male" else ["Mrs", "Miss", "Rare"]
  )
  title = st.selectbox("Unvan (Title)", title_options)
  age = st.slider("Yaş (Age)", 0.42, 80.0, 28.0, 1.0)

with col2:
  sibsp = st.number_input("Kardeş / Eş Sayısı (SibSp)", 0, 8, 0)
  parch = st.number_input("Ebeveyn / Çocuk Sayısı (Parch)", 0, 6, 0)
  cabin = st.text_input("Kabin Numarası (Opsiyonel)", "", help="Örn: C123")
  fare = st.slider("Bilet Ücreti (Fare)", 0.0, 500.0, 32.2, 1.0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Tahmin Et 🔍", use_container_width=True, type="primary"):
  payload = {
      "Pclass": pclass,
      "Sex": sex,
      "Title": title,
      "Age": age,
      "Fare": fare,
      "SibSp": sibsp,
      "Parch": parch,
      "Cabin": cabin if cabin.strip() != "" else None,
  }

  try:
    response = requests.post(
        "http://127.0.0.1:8000/titanic/predict", json=payload
    )
    if response.status_code == 200:
      result = response.json()
      survived = result["prediction"]
      probability = result["probability"]["survived"]
      st.divider()
      if survived == 1:
        st.success(
            f"🎉 **Bu yolcunun hayatta kalma ihtimali yüksek!** (Olasılık:"
            f" %{probability * 100:.2f})"
        )
      else:
        st.error(
            f"⚠️ **Maalesef bu yolcu için hayatta kalma ihtimali düşük.**"
            f" (Olasılık: %{probability * 100:.2f})"
        )
      st.progress(float(probability))
    else:
      st.error(f"API Hatası: {response.text}")
  except requests.exceptions.ConnectionError:
    st.error("⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor.")