import requests
import streamlit as st

# 1. EN ÖNEMLİ KURAL: st.set_page_config en başta çağrılmalı!
st.set_page_config(
    page_title="Multi-Model ML Platformu",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",  # <-- Sol menü başlangıçta kapalı gelir
)

# Sidebar buton tasarımı için özel CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] div.stButton > button {
        border: none;
        background-color: transparent;
        text-align: left;
        width: 100%;
        padding: 8px 10px;
        color: inherit;
        font-size: 16px;
        border-radius: 4px;
    }
    [data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(150, 150, 150, 0.15);
        border: none;
        color: inherit;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("Model Seçimi")

if "selected_model" not in st.session_state:
  st.session_state.selected_model = None

# Backend'den model listesini çek
try:
  response = requests.get("http://127.0.0.1:8000/models")
  if response.status_code == 200:
    raw_list = response.json().get("models", [])
    # Sadece model dosyalarını filtrele, yardımcı sütunları at
    model_list = [m for m in raw_list if "model_columns" not in m]
  else:
    model_list = []
except Exception:
  model_list = []
  st.sidebar.error("Backend bağlantısı kurulamadı!")

# Ana sayfaya dönme butonu
if st.sidebar.button("🏠 Ana Sayfa", key="btn_home"):
  st.session_state.selected_model = None

if model_list:
  st.sidebar.markdown("---")
  st.sidebar.markdown("**Mevcut Modeller:**")
  for model_name in model_list:
    if "titanic" in model_name:
      display_name = "🚢 Titanic Survival"
    elif "bankloan" in model_name:
      display_name = "💳 Bank Loan"
    elif "cancer_knn" in model_name or "knn" in model_name:
      display_name = "🎗️ Meme Kanseri (KNN)"
    elif "cancer" in model_name:
      display_name = "🎗️ Meme Kanseri (SVM)"
    elif "drug200" in model_name or "drug" in model_name:
      display_name = "💊 Drug200 Teşhisi"
    elif "student" in model_name:
      display_name = "🎓 Öğrenci Durum Analizi"
    else:
      display_name = model_name

    if st.sidebar.button(display_name, key=f"btn_{model_name}"):
      st.session_state.selected_model = model_name
else:
  st.sidebar.warning("models/ klasöründe model bulunamadı.")


# --- ANA EKRAN YÖNLENDİRMESİ ---

if st.session_state.selected_model is None:
  # --- 1. KARŞILAMA / LANDING EKRANI ---
  st.title("🚀 Makine Öğrenmesi Tahmin Platformu")
  st.markdown(
      "Hoş geldiniz! Bu platform, FastAPI tabanlı arka uç servisleri ve"
      " Scikit-Learn Pipeline yapıları üzerine kurulmuştur."
  )
  st.info(
      "👉 Başlamak için sol üstteki menü ikonuna tıklayıp **model seçimi**"
      " yapabilirsiniz."
  )

  st.markdown("### Sistemde Yer Alan Modeller:")
  col1, col2, col3, col4, col5, col6 = st.columns(6)
  with col1:
    st.markdown("#### 🚢 Titanic")
    st.write("Hayatta kalma.")
  with col2:
    st.markdown("#### 💳 Banka")
    st.write("Kredi onayı.")
  with col3:
    st.markdown("#### 🎗️ Kanser (SVM)")
    st.write("SVM teşhis.")
  with col4:
    st.markdown("#### 🎗️ Kanser (KNN)")
    st.write("KNN teşhis.")
  with col5:
    st.markdown("#### 💊 Drug200")
    st.write("İlaç önerisi.")
  with col6:
    st.markdown("#### 🎓 Öğrenci")
    st.write("Öğrenci analizi.")

elif "titanic" in st.session_state.selected_model:
  # --- 2. TITANIC FORMU ---
  st.title("🚢 Titanic Hayatta Kalma Tahmin Sistemi")
  st.markdown(
      "Yolcu bilgilerini girerek hayatta kalma ihtimalini anlık olarak"
      " hesaplayın."
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
          "http://127.0.0.1:8000/predict/titanic", json=payload
      )
      if response.status_code == 200:
        result = response.json()
        survived = result["survived"]
        probability = result["survival_probability"]
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
        st.error("API'den beklenmeyen bir hata döndü.")
    except requests.exceptions.ConnectionError:
      st.error(
          "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor gibi görünüyor."
      )

elif "bankloan" in st.session_state.selected_model:
  # --- 3. BANK LOAN FORMU ---
  st.title("💳 Banka Kredi Onay Tahmin Sistemi")
  st.markdown(
      "Müşteri detaylarını girerek kredi onay durumunu ve olasılığını hesaplayın."
  )
  st.divider()

  col1, col2 = st.columns(2)
  with col1:
    age_loan = st.number_input("Yaş (Age)", 18, 100, 35)
    experience = st.number_input("Deneyim Yılı (Experience)", 0, 50, 10)
    income = st.number_input("Yıllık Gelir ($Bin)", 10, 500, 85)
    family = st.number_input("Aile Büyüklüğü", 1, 4, 3)
    ccavg = st.number_input("Aylık Kredi Kartı Harcaması ($Bin)", 0.0, 20.0, 2.5)

  with col2:
    education = st.selectbox(
        "Eğitim Seviyesi",
        [1, 2, 3],
        format_func=lambda x: (
            "1 - Lisans"
            if x == 1
            else ("2 - Yüksek Lisans" if x == 2 else "3 - Doktora/İleri")
        ),
    )
    mortgage = st.number_input("Konut İpoteği ($Bin)", 0.0, 1000.0, 0.0)
    securities = st.selectbox(
        "Menkul Kıymet Hesabı", [0, 1], format_func=lambda x: "Var" if x == 1 else "Yok"
    )
    cd_account = st.selectbox(
        "Mevduat Sertifikası (CD)", [0, 1], format_func=lambda x: "Var" if x == 1 else "Yok"
    )
    online = st.selectbox(
        "İnternet Bankacılığı", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )
    credit_card = st.selectbox(
        "Banka Kredi Kartı", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )

  st.markdown("<br>", unsafe_allow_html=True)
  if st.button(
      "Krediyi Tahmin Et 🔍", use_container_width=True, type="primary"
  ):
    payload = {
        "Age": age_loan,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities_Account": securities,
        "CD_Account": cd_account,
        "Online": online,
        "Credit_Card": credit_card,
    }
    try:
      response = requests.post(
          "http://127.0.0.1:8000/predict/bankloan", json=payload
      )
      if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probability = result["probability"]
        st.divider()
        if prediction == 1:
          st.success(
              f"✅ **Kredi Onaylanabilir!** (Onay Olasılığı:"
              f" %{probability * 100:.2f})"
          )
        else:
          st.error(
              f"❌ **Kredi Reddedilebilir.** (Red Riski Yüksek, Olasılık:"
              f" %{probability * 100:.2f})"
          )
        st.progress(float(probability))
      else:
        st.error("API'den beklenmeyen bir hata döndü.")
    except requests.exceptions.ConnectionError:
      st.error(
          "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor gibi görünüyor."
      )

elif "cancer" in st.session_state.selected_model:
  # --- 4. CANCER DETECTION FORMU (SVM & KNN) ---
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

  is_knn = "knn" in st.session_state.selected_model

  if is_knn:
    st.title("🎗️ Meme Kanseri Teşhis Sistemi (KNN)")
    api_endpoint = "http://127.0.0.1:8000/predict/cancer_knn"
  else:
    st.title("🎗️ Meme Kanseri Teşhis Sistemi (SVM)")
    api_endpoint = "http://127.0.0.1:8000/predict/cancer"

  st.markdown(
      "Tümör ölçüm değerlerini girerek iyi huylu veya kötü huylu olma"
      " durumunu analiz edin."
  )
  st.divider()

  payload = {}
  col1, col2 = st.columns(2)

  for i, col_name in enumerate(features_list):
    target_col = col1 if i % 2 == 0 else col2
    with target_col:
      payload[col_name] = st.number_input(
          f"{col_name}",
          value=float(defaults.get(col_name, 0.0)),
          format="%.4f",
          key=f"{st.session_state.selected_model}_{col_name}",
      )

  st.markdown("<br>", unsafe_allow_html=True)
  if st.button("Teşhisi Başlat 🔍", use_container_width=True, type="primary"):
    try:
      response = requests.post(api_endpoint, json=payload)
      if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]  # 0: Benign, 1: Malignant
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
        st.error("API'den beklenmeyen bir hata döndü.")
    except requests.exceptions.ConnectionError:
      st.error(
          "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor gibi görünüyor."
      )

elif "drug" in st.session_state.selected_model:
  # --- 5. DRUG200 FORMU ---
  st.title("💊 Drug200 İlaç Sınıflandırma Sistemi")
  st.markdown("Hasta parametrelerini girerek uygun ilaç önerisini alın.")
  st.divider()

  col1, col2 = st.columns(2)
  with col1:
    age_drug = st.number_input("Yaş (Age)", 1, 100, 30)
    sex_drug = st.selectbox(
        "Cinsiyet (Sex)",
        [0, 1],
        format_func=lambda x: "Kadın (F)" if x == 0 else "Erkek (M)",
    )
    bp_drug = st.selectbox(
        "Kan Basıncı (BP)",
        [0, 1, 2],
        format_func=lambda x: (
            "Düşük (Low)"
            if x == 0
            else ("Normal" if x == 1 else "Yüksek (High)")
        ),
    )

  with col2:
    chol_drug = st.selectbox(
        "Kolesterol (Cholesterol)",
        [0, 1],
        format_func=lambda x: "Normal" if x == 0 else "Yüksek (High)",
    )
    na_to_k = st.number_input(
        "Sodyum / Potasyum Oranı (Na_to_K)", 0.0, 50.0, 15.0, format="%.2f"
    )

  st.markdown("<br>", unsafe_allow_html=True)
  if st.button("İlacı Tahmin Et 🔍", use_container_width=True, type="primary"):
    payload = {
        "Age": age_drug,
        "Sex": sex_drug,
        "BP": bp_drug,
        "Cholesterol": chol_drug,
        "Na_to_K": na_to_k,
    }
    try:
      response = requests.post(
          "http://127.0.0.1:8000/predict/drug", json=payload
      )
      if response.status_code == 200:
        result = response.json()
        prediction = result["prediction"]
        probabilities = result["probabilities"]

        st.divider()
        st.success(f"🎯 **Önerilen İlaç / Sınıf:** `{prediction}`")

        st.markdown("### 📊 Sınıf Olasılıkları:")
        for cls_name, prob in probabilities.items():
          st.write(f"- **{cls_name}**: %{prob * 100:.2f}")
          st.progress(float(prob))
      else:
        st.error("API'den beklenmeyen bir hata döndü.")
    except requests.exceptions.ConnectionError:
      st.error(
          "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor gibi görünüyor."
      )

elif "student" in st.session_state.selected_model:
  # --- 6. STUDENT FORMU ---
  st.title("🎓 Öğrenci Durum Analiz Sistemi")
  st.markdown(
      "Öğrenci performans parametrelerini girerek durum analizini yapın."
  )
  st.divider()

  col1, col2 = st.columns(2)
  with col1:
    study_hours = st.number_input(
        "Günlük Çalışma Saati (study_hours)", 0.0, 24.0, 4.0, format="%.1f"
    )
    attendance = st.number_input(
        "Devam Oranı (attendance)", 0.0, 100.0, 85.0, format="%.1f"
    )
    sleep_hours = st.number_input(
        "Uyku Saati (sleep_hours)", 0.0, 24.0, 7.0, format="%.1f"
    )
    internet_usage = st.number_input(
        "İnternet Kullanımı (internet_usage)", 0.0, 24.0, 3.0, format="%.1f"
    )

  with col2:
    assignments_completed = st.number_input(
        "Tamamlanan Ödev Sayısı (assignments_completed)",
        0.0,
        100.0,
        10.0,
        format="%.1f",
    )
    previous_score = st.number_input(
        "Önceki Puan (previous_score)", 0.0, 100.0, 75.0, format="%.1f"
    )
    exam_score = st.number_input(
        "Sınav Puanı (exam_score)", 0.0, 100.0, 70.0, format="%.1f"
    )

  st.markdown("<br>", unsafe_allow_html=True)
  if st.button(
      "Analizi Başlat 🔍", use_container_width=True, type="primary"
  ):
    # Modelin beklediği 7 sütun anahtarı
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
        probabilities = result["probabilities"]

        st.divider()
        st.success(f"🎯 **Tahmin Edilen Durum:** `{prediction}`")

        st.markdown("### 📊 Sınıf Olasılıkları:")
        for cls_name, prob in probabilities.items():
          st.write(f"- **{cls_name}**: %{prob * 100:.2f}")
          st.progress(float(prob))
      else:
        st.error(f"API Hatası: {response.text}")
    except requests.exceptions.ConnectionError:
      st.error(
          "⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor gibi görünüyor."
      )