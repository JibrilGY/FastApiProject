import os
import sys
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import create_model
from sklearn.base import BaseEstimator, TransformerMixin


# 1. Eksik verilerde hata vermeyen, güvenli Feature Engineer sınıfı
class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.age_median_ = None
    self.fare_median_ = None
    self.feature_names_in_ = None

  def fit(self, X, y=None):
    self.age_median_ = X["Age"].median() if "Age" in X.columns else 28.0
    self.fare_median_ = X["Fare"].median() if "Fare" in X.columns else 14.4
    self.feature_names_in_ = X.columns.tolist()
    return self

  def transform(self, X):
    df = X.copy()

    # Title Extraction (Name yoksa veya boşsa çökmesin, "Unknown" versin)
    if "Name" in df.columns and df["Name"].notna().any():
      df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
      df["Title"] = df["Title"].fillna("Unknown")
      df["Title"] = df["Title"].replace(
          [
              "Lady",
              "Countess",
              "Capt",
              "Col",
              "Don",
              "Dr",
              "Major",
              "Rev",
              "Sir",
              "Jonkheer",
              "Dona",
          ],
          "Rare",
      )
      df["Title"] = df["Title"].replace(
          {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
      )
    else:
      df["Title"] = "Unknown"

    # Family Size & IsAlone
    sibsp = (
        df["SibSp"] if "SibSp" in df.columns else pd.Series(0, index=df.index)
    )
    parch = (
        df["Parch"] if "Parch" in df.columns else pd.Series(0, index=df.index)
    )
    df["FamilySize"] = sibsp.fillna(0) + parch.fillna(0) + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Deck Extraction (Cabin yoksa "U" versin)
    if "Cabin" in df.columns and df["Cabin"].notna().any():
      df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0]
      df["Deck"] = df["Deck"].replace("T", "U")
    else:
      df["Deck"] = "U"

    # Eksik yaş ve ücretleri doldurma
    if "Age" in df.columns:
      df["Age"] = df["Age"].fillna(
          self.age_median_ if self.age_median_ is not None else 28.0
      )
    else:
      df["Age"] = 28.0

    if "Fare" in df.columns:
      df["Fare"] = df["Fare"].fillna(
          self.fare_median_ if self.fare_median_ is not None else 14.4
      )
    else:
      df["Fare"] = 14.4

    required_features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "FamilySize",
        "IsAlone",
        "Title",
        "Deck",
    ]

    # Garanti olması için eksik kalan kolon varsa oluşturalım
    for col in required_features:
      if col not in df.columns:
        df[col] = 0 if col != "Sex" else "male"

    return df[required_features]


# Uvicorn / Pickle Modül Eşlemesi
sys.modules["__main__"].TitanicFeatureEngineer = TitanicFeatureEngineer
if "__mp_main__" in sys.modules:
  sys.modules["__mp_main__"].TitanicFeatureEngineer = TitanicFeatureEngineer

router = APIRouter(prefix="/titanic", tags=["Titanic Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "titanic_pipeline.pkl")

pipeline = None
TitanicInput = create_model("TitanicInput")

try:
  pipeline = joblib.load(MODEL_PATH)
  raw_features = [
      "Pclass",
      "Sex",
      "Age",
      "Fare",
      "SibSp",
      "Parch",
      "Name",
      "Cabin",
  ]
  field_definitions = {field: (object, None) for field in raw_features}
  TitanicInput = create_model("TitanicInput", **field_definitions)
  print("✅ Titanic modeli ve girdi şeması başarıyla yüklendi!")

except Exception as e:
  print(f"❌ Titanic Yükleme Hatası: {e}")
  pipeline = None
  TitanicInput = create_model("TitanicInput")


@router.post("/predict")
def predict_titanic(data: TitanicInput):
  if not pipeline:
    raise HTTPException(
        status_code=500, detail="Titanic modeli yüklenemedi!"
    )

  try:
    input_df = pd.DataFrame([data.model_dump()])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0].tolist()

    return {
        "status": "success",
        "prediction": int(prediction),
        "probability": {
            "survived": round(probability[1], 4),
            "did_not_survive": round(probability[0], 4),
        },
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))