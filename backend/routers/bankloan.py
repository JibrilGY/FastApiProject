import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from schemas import StandardResponse, BankloanInput

router = APIRouter(prefix="/bankloan", tags=["Bankloan Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bankloan", "random_forest_model.pkl")

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Bankloan model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Bankloan Yükleme Hatası: {e}")
    model = None


@router.post("/predict", response_model=StandardResponse)
def predict_bankloan(data: BankloanInput):
    if not model:
        raise HTTPException(
            status_code=500, detail="Bankloan model yüklenemedi!"
        )

    try:
        input_df = pd.DataFrame([data.model_dump()])

        # Kolon adı eşleştirmesi (CD_Account -> CD.Account)
        rename_mapping = {
            "CD_Account": "CD.Account"
        }
        input_df = input_df.rename(columns=rename_mapping)

        # 🔑 KRİTİK ÇÖZÜM: Modelin eğitimde kullandığı kolon sırasını ve adlarını zorunlu tut
        if hasattr(model, "feature_names_in_"):
            input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()

        return StandardResponse(
            status="success",
            prediction=int(prediction),
            probability={
                "approved": round(probability[1], 4),
                "rejected": round(probability[0], 4),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))