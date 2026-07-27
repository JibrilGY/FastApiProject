import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A simple API to predict Titanic passenger survival using Advanced Pipeline",
    version="1.0",
)

# Pipeline modelini yüklüyoruz (Artık model_columns'a FastAPI tarafında gerek yok!)
pipeline_model = joblib.load("models/titanic_pipeline.pkl")


class PassengerInput(BaseModel):
  Pclass: int
  Sex: str
  Title: str
  Name: str
  Age: float
  Fare: float
  SibSp: int
  Parch: int
  Cabin: str | None = None


@app.post("/predict")
def predict_survival(data: PassengerInput):
  input_data = data.model_dump()

  # Cinsiyet standardizasyonu
  if input_data["Sex"].lower() in ["erkek", "male"]:
    input_data["Sex"] = "male"
  else:
    input_data["Sex"] = "female"

  df = pd.DataFrame([input_data])

  # Feature Engineering (Eğitim aşamasındaki kurallarla birebir aynı)
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
  df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0].replace("T", "U")

  # Modelin beklediği ham sütun listesi
  features = [
      "Pclass",
      "Sex",
      "Age",
      "Fare",
      "FamilySize",
      "IsAlone",
      "Title",
      "Deck",
  ]
  X = df[features].copy()

  # DİKKAT: Hiçbir manuel get_dummies veya reindex yapmıyoruz!
  # Tüm One-Hot, Polynomial ve Scaling işlemlerini pipeline kendi içinde yapıyor.
  prediction = pipeline_model.predict(X)[0]
  probability = pipeline_model.predict_proba(X)[0][1]

  return {
      "survived": int(prediction),
      "survival_probability": round(float(probability), 4),
  }