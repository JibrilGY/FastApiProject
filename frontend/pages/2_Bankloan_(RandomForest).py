import requests
import streamlit as st

st.title("💳 Bank Loan Approval Prediction System")
st.markdown("Enter customer details to evaluate loan eligibility and approval probability.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($k)", 10, 500, 85)
    ccavg = st.number_input("Avg. Monthly Credit Card Spend ($k)", 0.0, 20.0, 2.5)
    mortgage = st.number_input("House Mortgage Value ($k)", 0.0, 1000.0, 0.0)

with col2:
    education = st.selectbox(
        "Education Level",
        [1, 2, 3],
        format_func=lambda x: "1 - Undergrad" if x == 1 else ("2 - Graduate" if x == 2 else "3 - Advanced/Doctorate")
    )
    cd_account = st.selectbox(
        "Certificate of Deposit (CD) Account",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Loan Status 🔍", use_container_width=True, type="primary"):
    payload = {
        "Income": income,
        "CCAvg": ccavg,
        "Mortgage": mortgage,
        "Education": education,
        "CD_Account": cd_account
    }

    try:
        response = requests.post("http://127.0.0.1:8000/bankloan/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]
            probability = result["probability"]["approved"]

            st.divider()
            if prediction == 1:
                st.success(f"✅ **Loan Approved!** (Approval Probability: %{probability * 100:.2f})")
            else:
                st.error(f"❌ **Loan Rejected.** (High Risk, Approval Probability: %{probability * 100:.2f})")

            st.progress(float(probability))
        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Connection Error:** FastAPI server is not running.")