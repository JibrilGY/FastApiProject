import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import bankloan, cancer_knn, cancer_svm, drug, student, titanic

app = FastAPI(title="Multi-Model ML Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adding Routers
app.include_router(titanic.router)
app.include_router(bankloan.router)
app.include_router(cancer_svm.router)
app.include_router(cancer_knn.router)
app.include_router(drug.router)
app.include_router(student.router)


@app.get("/models")
def list_models():
  models_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "models"
  )
  if not os.path.exists(models_dir):
    return {"models": []}

  files = [
      f.replace(".pkl", "")
      for f in os.listdir(models_dir)
      if f.endswith(".pkl")
  ]
  return {"models": files}