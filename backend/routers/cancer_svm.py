import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from app_schemas import CancerInput, StandardResponse

router = APIRouter(prefix="/cancer-svm", tags=["Cancer SVM Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer", "svm_model.pkl")
BUNDLE_PATH = os.path.join(
    BASE_DIR, "models", "cancer", "cancer_data_bundle.pkl"
)

model = None
bundle = None

try:
  model = joblib.load(MODEL_PATH)
  bundle = joblib.load(BUNDLE_PATH)
  print("✅ Cancer model and bundle loaded successfully!")
except Exception as e:
  print(f"❌ Cancer Loading Error: {e}")
  model = None
  bundle = None


@router.post("/predict-svm", response_model=StandardResponse)
def predict_cancer(data: CancerInput):
  if not model or not bundle:
    raise HTTPException(
        status_code=500, detail="Cancer model or bundle could not be loaded!"
    )

  try:
    scaler = bundle["scaler"]
    selected_features = bundle["selected_features"]
    pt = bundle["pt"]

    # 1. Convert incoming data to DataFrame
    input_dict = data.model_dump(by_alias=True)
    input_df = pd.DataFrame([input_dict])

    # 2. Derive the radius_growth_ratio feature required by the model
    if "radius_worst" in input_df.columns and "radius_mean" in input_df.columns:
      input_df["radius_growth_ratio"] = input_df["radius_worst"] / (
          input_df["radius_mean"] + 1e-5
      )

    # 3. If the pt object expects all original columns, create the full matrix by filling missing ones with 0
    if hasattr(pt, "feature_names_in_"):
      full_df = pd.DataFrame(columns=pt.feature_names_in_)
      for col in input_df.columns:
        if col in full_df.columns:
          full_df.loc[0, col] = input_df.loc[0, col]
      # Fill remaining raw columns (not requested from the user) with 0
      full_df = full_df.fillna(0)
      transformed_array = pt.transform(full_df)
      transformed_df = pd.DataFrame(transformed_array, columns=pt.feature_names_in_)
    else:
      transformed_array = pt.transform(input_df)
      transformed_df = pd.DataFrame(transformed_array, columns=input_df.columns)

    # 4. Filter the 10 features selected via ANOVA
    selected_df = transformed_df[selected_features]

    # 5. Scale using StandardScaler
    scaled_array = scaler.transform(selected_df)

    # 6. Prediction and Probability Calculation
    prediction = model.predict(scaled_array)[0]
    probability = model.predict_proba(scaled_array)[0].tolist()

    return StandardResponse(
        status="success",
        prediction=int(prediction),
        probability={
            "malignant": round(probability[1], 4),  # 1: Malignant
            "benign": round(probability[0], 4),     # 0: Benign
        },
    )

  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))