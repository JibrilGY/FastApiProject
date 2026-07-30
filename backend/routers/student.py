import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas import StandardResponse, StudentInput

router = APIRouter(prefix="/student", tags=["Student Placement Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "student", "xg_model.pkl")

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Student model loaded successfully!")
except Exception as e:
    print(f"❌ Student Loading Error: {e}")
    model = None


@router.post("/predict", response_model=StandardResponse)
def predict_student(data: StudentInput):
    if not model:
        raise HTTPException(
            status_code=500, detail="Student model could not be loaded!"
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])

        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()

        return StandardResponse(
            status="success",
            prediction=int(prediction),
            probability={
                "placed": round(probability[1], 4),
                "not_placed": round(probability[0], 4),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))