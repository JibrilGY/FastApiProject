import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from app_schemas import DrugInput

router = APIRouter(prefix="/drug", tags=["Drug Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drug", "decision_tree_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "drug", "encoders_dict.pkl")
COLS_PATH = os.path.join(BASE_DIR, "models", "drug", "categorical_cols.pkl")

model = None
encoders = None
categorical_cols = None

try:
  model = joblib.load(MODEL_PATH)
  encoders = joblib.load(ENCODERS_PATH)
  categorical_cols = joblib.load(COLS_PATH)
  print("✅ Drug model and dynamic encoders successfully loaded!")
except Exception as e:
  print(f"❌ Drug Model/Encoder Loading Error: {e}")


@router.post("/predict")
def predict_drug(data: DrugInput):
  if model is None or encoders is None:
    raise HTTPException(
        status_code=500, detail="Drug model or encoders could not be loaded."
    )

  try:
    input_df = pd.DataFrame([data.model_dump()])

    if "Sex" in input_df.columns:
      input_df = input_df.drop(columns=["Sex"])

    for col in categorical_cols:
      if col in input_df.columns:
        encoder = encoders[col]
        input_df[[col]] = encoder.transform(input_df[[col]])

    if hasattr(model, "feature_names_in_"):
      input_df = input_df[model.feature_names_in_]

    prediction = model.predict(input_df)

    return {
        "model": "Drug Decision Tree Pipeline",
        "prediction": str(prediction[0]),
        "status": "success",
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))