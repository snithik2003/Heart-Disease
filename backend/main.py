import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "heart_disease_model.pkl"))

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

model = None
load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    load_error = str(e)

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Logistic Regression model trained on the UCI Cleveland Heart Disease dataset. "
                "Educational project, not a medical device.",
    version="1.0.0",
)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type: 1 typical angina, 2 atypical angina, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl: 1 yes, 0 no")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy")
    thalach: float = Field(..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise induced angina: 1 yes, 0 no")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise relative to rest")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST segment: 1 upsloping, 2 flat, 3 downsloping")
    ca: Literal[0, 1, 2, 3] = Field(..., description="Number of major vessels colored by fluoroscopy (0-3)")
    thal: Literal[3, 6, 7] = Field(..., description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible defect")

    model_config = {
        "json_schema_extra": {
            "example": {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
                        "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5,
                        "slope": 2, "ca": 3, "thal": 3}
        }
    }


class Prediction(BaseModel):
    prediction: int
    result: str
    probability: dict


@app.get("/")
def root():
    return {
        "message": "Welcome to the Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
        "web_interface": "/app/",
        "disclaimer": "Educational project only. Not a medical diagnosis.",
    }


@app.get("/health")
def health():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "error", "model_loaded": False, "detail": load_error})
    return {"status": "ok", "model_loaded": True, "scikit_learn_version": sklearn.__version__}


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    df = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    p = dict(zip([int(c) for c in model.classes_], proba))
    return {
        "prediction": pred,
        "result": "Heart Disease Detected" if pred == 1 else "No Heart Disease Detected",
        "probability": {
            "no_heart_disease": round(float(p[0]), 4),
            "heart_disease": round(float(p[1]), 4),
        },
    }


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error while predicting."})


for candidate in (BASE_DIR / "frontend", BASE_DIR.parent / "frontend"):
    if candidate.is_dir():
        app.mount("/app", StaticFiles(directory=candidate, html=True), name="frontend")
        break
