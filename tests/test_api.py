import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from main import app, FEATURES  # noqa: E402

client = TestClient(app)

HIGH_RISK = {"age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
             "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3}
LOW_RISK = {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0, "restecg": 2,
            "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome" in r.json()["message"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


@pytest.mark.parametrize("patient", [HIGH_RISK, LOW_RISK])
def test_predict_matches_saved_model(patient):
    r = client.post("/predict", json=patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    probs = body["probability"]
    assert abs(probs["no_heart_disease"] + probs["heart_disease"] - 1) < 1e-3

    model = joblib.load(BACKEND / "heart_disease_model.pkl")
    df = pd.DataFrame([patient])[FEATURES].astype(float)
    assert body["prediction"] == int(model.predict(df)[0])
    assert abs(probs["heart_disease"] - model.predict_proba(df)[0][1]) < 1e-3


def test_high_risk_scores_higher_than_low_risk():
    hi = client.post("/predict", json=HIGH_RISK).json()["probability"]["heart_disease"]
    lo = client.post("/predict", json=LOW_RISK).json()["probability"]["heart_disease"]
    assert hi > lo


def test_missing_field():
    bad = dict(HIGH_RISK)
    del bad["age"]
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range_value():
    assert client.post("/predict", json={**HIGH_RISK, "age": 500}).status_code == 422


def test_invalid_category():
    assert client.post("/predict", json={**HIGH_RISK, "cp": 9}).status_code == 422


def test_wrong_type():
    assert client.post("/predict", json={**HIGH_RISK, "chol": "abc"}).status_code == 422
