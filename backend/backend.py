import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, create_model

app = FastAPI(
    title="Multi-Model Prediction API",
    description="Pipeline destekli çoklu model tahmin servisi",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"


@app.get("/models")
def get_available_models():
  if not os.path.exists(MODELS_DIR):
    return {"models": []}
  files = os.listdir(MODELS_DIR)
  models = [f.replace(".pkl", "") for f in files if f.endswith(".pkl")]
  return {"models": models}


# --- Modelleri Güvenle Yükle ---
try:
  titanic_pipeline = joblib.load(
      os.path.join(MODELS_DIR, "titanic_pipeline.pkl")
  )
except Exception:
  titanic_pipeline = None

try:
  bankloan_pipeline = joblib.load(
      os.path.join(MODELS_DIR, "bankloan_pipeline.pkl")
  )
  bankloan_columns = joblib.load(
      os.path.join(MODELS_DIR, "model_columns_random_forest.pkl")
  )
except Exception:
  bankloan_pipeline = None
  bankloan_columns = None

try:
  cancer_data_pipeline = joblib.load(
      os.path.join(MODELS_DIR, "cancer_data_pipeline.pkl")
  )
except Exception:
  cancer_data_pipeline = None

try:
  cancer_data_pipeline_knn = joblib.load(
      os.path.join(MODELS_DIR, "cancer_data_pipeline_knn.pkl")
  )
except Exception:
  cancer_data_pipeline_knn = None

try:
  drug200_pipeline = joblib.load(
      os.path.join(MODELS_DIR, "drug200_pipeline.pkl")
  )
  drug200_columns = joblib.load(
      os.path.join(MODELS_DIR, "model_columns_decision_tree.pkl")
  )
except Exception:
  drug200_pipeline = None
  drug200_columns = None

try:
  student_pipeline = joblib.load(
      os.path.join(MODELS_DIR, "student_pipeline.pkl")
  )
  student_columns = joblib.load(
      os.path.join(MODELS_DIR, "model_columns_xgboost.pkl")
  )
except Exception:
  student_pipeline = None
  student_columns = None


# --- Pydantic Şemaları ---
class PassengerInput(BaseModel):
  Pclass: int
  Sex: str
  Title: str
  Age: float
  Fare: float
  SibSp: int
  Parch: int
  Cabin: str | None = None


class LoanInput(BaseModel):
  Age: int
  Experience: int
  Income: float
  Family: int
  CCAvg: float
  Education: int
  Mortgage: float
  Securities_Account: int
  CD_Account: int
  Online: int
  Credit_Card: int


# --- Dinamik Şemalar (Feature Names In) ---
if cancer_data_pipeline is not None:
  cancer_expected_cols = (
      cancer_data_pipeline.named_steps["preprocessor"].feature_names_in_
  )
  field_definitions = {col: (float, ...) for col in cancer_expected_cols}
  CancerInput = create_model("CancerInput", **field_definitions)

if cancer_data_pipeline_knn is not None:
  cancer_expected_cols_knn = (
      cancer_data_pipeline_knn.named_steps["preprocessor"].feature_names_in_
  )
  field_definitions = {col: (float, ...) for col in cancer_expected_cols_knn}
  CancerInput_KNN = create_model("CancerInput_KNN", **field_definitions)

if drug200_pipeline is not None:
  try:
    drug200_expected_cols = (
        drug200_pipeline.named_steps["preprocessor"].feature_names_in_
    )
  except Exception:
    drug200_expected_cols = drug200_columns
  field_definitions = {col: (float, ...) for col in drug200_expected_cols}
  DrugInput = create_model("DrugInput", **field_definitions)

# Student (XGBoost) için dinamik şema ve sütun eşitleme
if student_pipeline is not None:
  try:
    student_expected_cols = (
        student_pipeline.named_steps["preprocessor"].feature_names_in_
    )
  except Exception:
    student_expected_cols = student_columns
  field_definitions = {col: (float, ...) for col in student_expected_cols}
  StudentInput = create_model("StudentInput", **field_definitions)
else:
  student_expected_cols = []
  StudentInput = create_model("StudentInput")


# --- Endpoints ---
@app.post("/predict/titanic")
def predict_titanic(data: PassengerInput):
  if titanic_pipeline is None:
    raise HTTPException(status_code=404, detail="Titanic modeli bulunamadı.")

  input_data = data.model_dump()
  if input_data["Sex"].lower() in ["erkek", "male"]:
    input_data["Sex"] = "male"
  else:
    input_data["Sex"] = "female"

  df = pd.DataFrame([input_data])
  df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
  df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
  df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0].replace("T", "U")

  features = [
      "Pclass",
      "Sex",
      "Age",
      "Fare",
      "FamilySize",
      "IsAlone",
      "Title",
      "Deck",
  ]
  X = df[features].copy()

  prediction = int(titanic_pipeline.predict(X)[0])
  probability = float(titanic_pipeline.predict_proba(X)[0][1])

  return {
      "survived": prediction,
      "survival_probability": round(probability, 4),
  }


@app.post("/predict/bankloan")
def predict_bankloan(data: LoanInput):
  if bankloan_pipeline is None or bankloan_columns is None:
    raise HTTPException(status_code=404, detail="Bank Loan modeli bulunamadı.")

  input_data = data.model_dump()

  input_data["CD.Account"] = input_data.pop("CD_Account")
  input_data["Securities.Account"] = input_data.pop("Securities_Account")
  input_data["CreditCard"] = input_data.pop("Credit_Card")

  df = pd.DataFrame([input_data])
  X = df[bankloan_columns].copy()

  prediction = int(bankloan_pipeline.predict(X)[0])
  probabilities = bankloan_pipeline.predict_proba(X)[0]

  class_index = list(bankloan_pipeline.classes_).index(1)
  probability = float(probabilities[class_index])

  result_text = "Kredi Onaylanabilir" if prediction == 1 else "Kredi Reddedilebilir"

  return {
      "model": "bankloan",
      "prediction": prediction,
      "result_text": result_text,
      "probability": round(probability, 4),
  }


@app.post("/predict/cancer")
def predict_cancer(data: CancerInput):
  if cancer_data_pipeline is None:
    raise HTTPException(status_code=404, detail="Cancer modeli bulunamadı.")

  input_df_cancer = pd.DataFrame([data.model_dump()])
  input_df_cancer = input_df_cancer[cancer_expected_cols]

  prediction = int(cancer_data_pipeline.predict(input_df_cancer)[0])
  probability = cancer_data_pipeline.predict_proba(input_df_cancer)[0].tolist()

  return {
      "prediction": prediction,
      "probability_benign": round(probability[0], 4),
      "probability_malignant": round(probability[1], 4),
  }


@app.post("/predict/cancer_knn")
def predict_cancer_knn(data: CancerInput_KNN):
  if cancer_data_pipeline_knn is None:
    raise HTTPException(status_code=404, detail="Cancer KNN modeli bulunamadı.")

  input_df_cancer_knn = pd.DataFrame([data.model_dump()])
  input_df_cancer_knn = input_df_cancer_knn[cancer_expected_cols_knn]

  prediction = int(cancer_data_pipeline_knn.predict(input_df_cancer_knn)[0])
  probability = cancer_data_pipeline_knn.predict_proba(
      input_df_cancer_knn
  )[0].tolist()

  return {
      "prediction": prediction,
      "probability_benign": round(probability[0], 4),
      "probability_malignant": round(probability[1], 4),
  }


@app.post("/predict/drug")
def predict_drug(data: DrugInput):
  if drug200_pipeline is None:
    raise HTTPException(status_code=404, detail="Drug200 modeli bulunamadı.")

  input_df_drug = pd.DataFrame([data.model_dump()])
  input_df_drug = input_df_drug[drug200_expected_cols]

  prediction = str(drug200_pipeline.predict(input_df_drug)[0])
  probabilities = drug200_pipeline.predict_proba(input_df_drug)[0].tolist()
  classes = drug200_pipeline.classes_.tolist()

  prob_dict = {
      str(cls): round(prob, 4) for cls, prob in zip(classes, probabilities)
  }

  return {
      "prediction": prediction,
      "probabilities": prob_dict,
  }


@app.post("/predict/student")
def predict_student(data: StudentInput):
  if student_pipeline is None:
    raise HTTPException(status_code=404, detail="Student modeli bulunamadı.")

  input_df_student = pd.DataFrame([data.model_dump()])
  input_df_student = input_df_student[student_expected_cols]

  prediction = str(student_pipeline.predict(input_df_student)[0])
  probabilities = student_pipeline.predict_proba(input_df_student)[0].tolist()
  classes = student_pipeline.classes_.tolist()

  prob_dict = {
      str(cls): round(prob, 4) for cls, prob in zip(classes, probabilities)
  }

  return {
      "prediction": prediction,
      "probabilities": prob_dict,
  }