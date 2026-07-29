import requests
import streamlit as st

st.title("💳 Banka Kredi Onay Tahmin Sistemi")
st.markdown("Müşteri detaylarını girerek kredi onay durumunu ve olasılığını hesaplayın.")
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
        format_func=lambda x: "1 - Lisans" if x == 1 else ("2 - Yüksek Lisans" if x == 2 else "3 - Doktora/İleri")
    )
    mortgage = st.number_input("Konut İpoteği ($Bin)", 0.0, 1000.0, 0.0)
    securities = st.selectbox("Menkul Kıymet Hesabı", [0, 1], format_func=lambda x: "Var" if x == 1 else "Yok")
    cd_account = st.selectbox("Mevduat Sertifikası (CD)", [0, 1], format_func=lambda x: "Var" if x == 1 else "Yok")
    online = st.selectbox("İnternet Bankacılığı", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
    credit_card = st.selectbox("Banka Kredi Kartı", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("Krediyi Tahmin Et 🔍", use_container_width=True, type="primary"):
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
        "Credit_Card": credit_card
    }

    try:
        response = requests.post("http://127.0.0.1:8000/bankloan/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["probability"]["approved"]
            st.divider()
            if prediction == 1:
                st.success(f"✅ **Kredi Onaylanabilir!** (Onay Olasılığı: %{probability * 100:.2f})")
            else:
                st.error(f"❌ **Kredi Reddedilebilir.** (Red Riski Yüksek, Onay Olasılığı: %{probability * 100:.2f})")
            st.progress(float(probability))
        else:
            st.error(f"API Hatası: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Bağlantı Hatası:** FastAPI sunucusu çalışmıyor.")