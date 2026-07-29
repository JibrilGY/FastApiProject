import os
import sys
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import create_model
from sklearn.base import BaseEstimator, TransformerMixin


# 1. Notebook'taki özel BankloanCleaner sınıfı
class BankloanCleaner(BaseEstimator, TransformerMixin):

  def __init__(self):
    self.feature_names_in_ = None

  def fit(self, X, y=None):
    self.feature_names_in_ = X.columns.tolist()
    return self

  def transform(self, X):
    df = X.copy()
    if "Experience" in df.columns:
      df["Experience"] = df["Experience"].apply(lambda x: max(0, x))
    return df


# 2. Pickle/Uvicorn modül bağlama
sys.modules["__main__"].BankloanCleaner = BankloanCleaner
if "__mp_main__" in sys.modules:
  sys.modules["__mp_main__"].BankloanCleaner = BankloanCleaner

router = APIRouter(prefix="/bankloan", tags=["Bankloan Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bankloan_pipeline.pkl")

pipeline = None
BankloanInput = create_model("BankloanInput")

try:
  pipeline = joblib.load(MODEL_PATH)

  raw_features = [
      "Age",
      "Experience",
      "Income",
      "Family",
      "CCAvg",
      "Education",
      "Mortgage",
      "Securities_Account",
      "CD_Account",
      "Online",
      "Credit_Card",
  ]
  field_definitions = {field: (object, None) for field in raw_features}
  BankloanInput = create_model("BankloanInput", **field_definitions)
  print("✅ Bankloan pipeline modeli başarıyla yüklendi!")

except Exception as e:
  print(f"❌ Bankloan Yükleme Hatası: {e}")
  pipeline = None
  BankloanInput = create_model("BankloanInput")


@router.post("/predict")
def predict_bankloan(data: BankloanInput):
  if not pipeline:
    raise HTTPException(
        status_code=500, detail="Bankloan pipeline modeli yüklenemedi!"
    )

  try:
    input_df = pd.DataFrame([data.model_dump()])

    # ⚠️ Modelin beklediği nokta içeren orijinal kolon isimlerine dönüştürüyoruz
    rename_mapping = {
        "Securities_Account": "Securities.Account",
        "CD_Account": "CD.Account",
        "Credit_Card": "CreditCard",
    }
    input_df = input_df.rename(columns=rename_mapping)

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0].tolist()

    return {
        "status": "success",
        "prediction": int(prediction),
        "probability": {
            "approved": round(probability[1], 4),
            "rejected": round(probability[0], 4),
        },
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))