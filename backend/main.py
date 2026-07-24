import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A simple API to predict Titanic passenger survival using Advanced Pipeline",
    version="1.0"
)

# 1. YENİ MODELİ YÜKLE (Artık scaler ayrı değil, tüm işlemler pipeline içinde!)
pipeline_model = joblib.load("models/titanic_pipeline.pkl")
model_columns = joblib.load("models/model_columns.pkl")


# API'ye dışarıdan gelecek veri için Pydantic modeli
class PassengerInput(BaseModel):
    Pclass: int
    Sex: str  # 'male' veya 'female'
    Title: str  # YENİ EKLENDİ: Unvanı (Mr, Mrs, Miss vb.) artık arayüzden ayrı alıyoruz
    Name: str  # İsmi tutuyoruz ama arka planda model için kullanmayacağız
    Age: float
    Fare: float
    SibSp: int
    Parch: int
    Cabin: str | None = None


@app.post("/predict")
def predict_survival(data: PassengerInput):
    # 1. Gelen veriyi DataFrame'e dönüştürme
    input_data = data.model_dump()

    # --- KRİTİK DÜZELTME: Arayüzden gelen Türkçe cinsiyeti İngilizceye çevir ---
    if input_data["Sex"].lower() in ["erkek", "male"]:
        input_data["Sex"] = "male"
    else:
        input_data["Sex"] = "female"
    # -----------------------------------------------------------------------

    df = pd.DataFrame([input_data])

    # 2. Feature Engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Cabin bilgisinden Deck türetme
    df['Deck'] = df['Cabin'].fillna('U').astype(str).str[0].replace('T', 'U')

    # Modelin eğitimde kullandığı temel sütunlar
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone', 'Title', 'Deck']
    X = df[features].copy()

    # 3. One-Hot Encoding uygulama
    X = pd.get_dummies(X)

    # 4. Sütun uyumsuzluklarını önlemek için eğitimdeki sütun sırasına eşitleme
    X = X.reindex(columns=model_columns, fill_value=0)

    # 5. TAHMİN YAPMA
    prediction = pipeline_model.predict(X)[0]
    probability = pipeline_model.predict_proba(X)[0][1]

    return {
        "survived": int(prediction),
        "survival_probability": round(float(probability), 4)
    }