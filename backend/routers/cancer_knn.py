import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import create_model

router = APIRouter(prefix="/predict", tags=["Cancer KNN Model"])

# Path to the cancer knn pipeline model file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_data_pipeline_knn.pkl")

# Load the model and create a dynamic schema using feature_names_in_
try:
  model = joblib.load(MODEL_PATH)
  expected_cols = model.named_steps["preprocessor"].feature_names_in_
  field_definitions = {col: (float, ...) for col in expected_cols}
  DynamicCancerKNNInput = create_model("CancerKNNInput", **field_definitions)
except Exception as e:
  model = None
  DynamicCancerKNNInput = create_model("CancerKNNInput")


@router.post("/cancer_knn")
def predict_cancer_knn(data: DynamicCancerKNNInput):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Cancer model could not be loaded."
    )

  try:
    # Convert Pydantic data to a DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Perform prediction
    prediction = model.predict(input_df)

    return {
        "model": "CancerKNN Pipeline",
        "prediction": int(prediction[0]),
        "status": "success",
    }
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))