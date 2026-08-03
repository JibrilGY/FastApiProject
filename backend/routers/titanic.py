import os
import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from app_schemas import TitanicInput

router = APIRouter(prefix="/titanic", tags=["Titanic Model"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_PATH = os.path.join(
    BASE_DIR, "models", "titanic", "titanic_data_bundle.pkl"
)
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "titanic", "logistic_regression_model.pkl"
)

bundle = None
model = None

try:
    bundle = joblib.load(BUNDLE_PATH)
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    print("✅ Titanic data bundle and model successfully loaded!")
except Exception as e:
    print(f"❌ Titanic Loading Error: {e}")


def preprocess_input(data_dict: dict, bundle: dict) -> pd.DataFrame:
    df = pd.DataFrame([data_dict])

    # 1. Varsayılan Değer Atamaları
    df["Title"] = df["Title"].fillna("Mr")
    df["Cabin"] = df["Cabin"].fillna("None")
    df["Ticket"] = df["Ticket"].fillna("None")

    df["Cabin_Deck"] = (
        df["Cabin"].str[0].fillna("U")
        if "Cabin" in df.columns and df["Cabin"].notna().any()
        else "U"
    )

    # Gereksiz sütunları düşür
    df = df.drop(
        columns=[col for col in ["Cabin", "Ticket"] if col in df.columns]
    )

    # 2. Eksik Değer Doldurma (Eğitim Seti İstatistikleri ile)
    title_medians = bundle.get("title_medians")
    if title_medians is not None:
        df["Age"] = df["Age"].fillna(df["Title"].map(title_medians))
    df["Age"] = df["Age"].fillna(bundle.get("age_median", 28.0))
    df["Embarked"] = df["Embarked"].fillna(bundle.get("embarked_mode", "S"))

    for col in ["Fare", "SibSp", "Parch", "Pclass"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3. Kategorik Kodlama (Sadece bundle içindeki pre-fitted encoders kullanılıyor)
    encoders = bundle.get("encoders", {})
    encoded_list = []
    cat_cols = [col for col in encoders.keys() if col in df.columns]

    for col in cat_cols:
        encoder = encoders[col]
        arr = encoder.transform(df[[col]])
        encoded_df = pd.DataFrame(
            arr, columns=encoder.get_feature_names_out([col]), index=df.index
        )
        encoded_list.append(encoded_df)

    # Orijinal kategorik sütunları düşürüp encoder çıktılarını ekle
    df_encoded = pd.concat(
        [
            df.drop(columns=cat_cols, errors="ignore"),
            pd.concat(encoded_list, axis=1)
            if encoded_list
            else pd.DataFrame(index=df.index),
        ],
        axis=1,
    )

    # (NOT: Tek satırlık inference'da pd.get_dummies drop_first hataya yol açtığı için kaldırıldı)

    # 4. Yeo-Johnson Transformation (Eğer bundle içinde mevcutsa)
    numerical_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    if bundle.get("power_transformer") and all(
        col in df_encoded.columns for col in numerical_cols
    ):
        df_encoded[numerical_cols] = bundle["power_transformer"].transform(
            df_encoded[numerical_cols]
        )

    # 5. Modelin Seçtiği Top 5 Özellik ile Kesin Hizalama (Reindex)
    selected_features = bundle["selected_features"]
    df_aligned = df_encoded.reindex(columns=selected_features, fill_value=0)

    # 6. StandardScaler ile Ölçeklendirme
    scaled_array = bundle["scaler"].transform(df_aligned)
    X_final = pd.DataFrame(scaled_array, columns=selected_features)

    return X_final


@router.post("/predict")
def predict_titanic(data: TitanicInput):
    if not bundle or not model:
        raise HTTPException(
            status_code=500, detail="Titanic bundle or model could not be loaded!"
        )

    try:
        processed_df = preprocess_input(data.model_dump(), bundle)
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0].tolist()
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