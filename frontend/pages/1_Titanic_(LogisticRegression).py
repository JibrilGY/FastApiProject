import requests
import streamlit as st

st.title("🚢 Titanic Survival Prediction System")
st.markdown(
    "Instantly calculate survival probability based on key passenger features."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox(
        "Sex",
        ["male", "female"],
        format_func=lambda x: "Male" if x == "male" else "Female",
    )

    # Cinsiyete göre akıllı ünvan (Title) seçimi
    if sex == "male":
        title_options = ["Mr", "Master", "Rare"]
    else:
        title_options = ["Mrs", "Miss", "Rare"]

    title = st.selectbox(
        "Passenger Title (Ünvan)",
        title_options,
    )

with col2:
    fare = st.slider("Fare (Ücret)", 0.0, 500.0, 32.2, 1.0)

    cabin = st.text_input(
        "Cabin Number (Deck Extraction)",
        "",
        help="e.g., C123 (Leave blank for Unknown/U deck)",
    )

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict 🔍", use_container_width=True, type="primary"):
    payload = {
        "Pclass": 1,
        "Title": title,
        "Sex": sex,
        "Age": 28.0,
        "SibSp": 0,
        "Parch": 0,
        "Ticket": "None",
        "Fare": fare,
        "Cabin": cabin if cabin.strip() != "" else "None",
        "Embarked": "S",
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
                    f"🎉 **High survival probability for this passenger!** (Probability: {probability * 100:.2f}%)"
                )
            else:
                st.error(
                    f"⚠️ **Low survival probability for this passenger.** (Probability: {probability * 100:.2f}%)"
                )
            st.progress(float(probability))
        else:
            st.error(f"API Error: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("⚠️ **Connection Error:** FastAPI server is not running.")