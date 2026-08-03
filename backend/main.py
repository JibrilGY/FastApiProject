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
    return {"models": [], "count": 0}

  files = []
  for root, dirs, filenames in os.walk(models_dir):
    for f in filenames:
      if f.endswith(".pkl"):
        lower_name = f.lower()
        if not any(
            keyword in lower_name
            for keyword in ["bundle", "scaler", "encoder", "pt", "cols"]
        ):
          files.append(f.replace(".pkl", ""))

  return {"models": files, "count": len(files)}