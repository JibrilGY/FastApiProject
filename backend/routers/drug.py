import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/drug", tags=["Drug Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drug200_pipeline.pkl")

model = None
try:
  model = joblib.load(MODEL_PATH)
  print("✅ Drug modeli başarıyla yüklendi!")
except Exception as e:
  print(f"❌ Drug Model Yükleme Hatası: {e}")
  model = None


class DrugInput(BaseModel):
  Age: int
  Sex: str
  BP: str
  Cholesterol: str
  Na_to_K: float


@router.post("/predict")
def predict_drug(data: DrugInput):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Drug model could not be loaded."
    )

  try:
    input_df = pd.DataFrame([data.model_dump()])

    # ⚠️ Kriterlere göre sayısal dönüşüm (Notebook'taki Encoding mantığına göre ayarlayabilirsin)
    # Cinsiyet Dönüşümü
    if "Sex" in input_df.columns:
      input_df["Sex"] = input_df["Sex"].map({"M": 0, "F": 1})

    # Kan Basıncı (BP) Dönüşümü
    if "BP" in input_df.columns:
      # Örnek: LOW=0, NORMAL=1, HIGH=2 (veya alfabetik/sıralı farklı bir mapping)
      input_df["BP"] = input_df["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})

    # Kolesterol Dönüşümü
    if "Cholesterol" in input_df.columns:
      # Örnek: NORMAL=0, HIGH=1
      input_df["Cholesterol"] = input_df["Cholesterol"].map(
          {"NORMAL": 0, "HIGH": 1}
      )

    prediction = model.predict(input_df)

    return {
        "model": "Drug Pipeline",
        "prediction": str(prediction[0]),
        "status": "success",
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))