import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import create_model

router = APIRouter(prefix="/predict", tags=["Cancer SVM Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_data_pipeline_svm.pkl")

try:
  model = joblib.load(MODEL_PATH)
  expected_cols = model.named_steps["preprocessor"].feature_names_in_
  field_definitions = {col: (float, ...) for col in expected_cols}
  DynamicCancerSVMInput = create_model("CancerSVMInput", **field_definitions)
except Exception as e:
  model = None
  DynamicCancerSVMInput = create_model("CancerSVMInput")


@router.post("/cancer_svm")
def predict_cancer_svm(data: DynamicCancerSVMInput):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Cancer model could not be loaded."
    )

  try:
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)

    response = {
        "model": "CancerSVM Pipeline",
        "prediction": int(prediction[0]),
        "status": "success",
    }

    # Olasılık değerlerini ekleyelim (0: Benign/İyi, 1: Malignant/Kötü)
    if hasattr(model, "predict_proba"):
      probability = model.predict_proba(input_df)[0].tolist()
      response["probability_benign"] = round(probability[0], 4)
      response["probability_malignant"] = round(probability[1], 4)
    else:
      # Model predict_proba desteklemiyorsa tahmine göre varsayılan atama
      response["probability_benign"] = (
          1.0 if int(prediction[0]) == 0 else 0.0
      )
      response["probability_malignant"] = (
          1.0 if int(prediction[0]) == 1 else 0.0
      )

    return response
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))