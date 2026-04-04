# tests/test_app.py
# Experiment 2: Automated tests run in GitHub Actions
# Run: pytest tests/ -v
# Uses mocks so tests pass BEFORE model is trained (CI safe)

import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath("."))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# ── Mock model & metadata ──────────────────────────────────────────────────
_mock_model = MagicMock()
_mock_model.predict.return_value       = np.array([1])
_mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])

_mock_meta = {
    "run_id":     "ci-test-run-001",
    "model_type": "RandomForest",
    "features":   ["CGPA","Internships","Projects",
                   "AptitudeTestScore","SoftSkillsRating","SSC_Marks","HSC_Marks"],
    "metrics":    {"accuracy": 0.91, "f1": 0.89,
                   "precision": 0.88, "recall": 0.90, "roc_auc": 0.95}
}

import app as app_module
app_module._model    = _mock_model
app_module._metadata = _mock_meta

from app import app
client = TestClient(app)

SAMPLE_STUDENT = {
    "CGPA": 8.5, "Internships": 2, "Projects": 3,
    "AptitudeTestScore": 82, "SoftSkillsRating": 4.2,
    "SSC_Marks": 80, "HSC_Marks": 76
}

# ── Tests ──────────────────────────────────────────────────────────────────
def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_returns_placed_or_not():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=SAMPLE_STUDENT)
    assert r.status_code == 200
    assert r.json()["placement_status"] in ["Placed", "Not Placed"]

def test_predict_probability_range():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=SAMPLE_STUDENT)
    prob = r.json()["probability_placed"]
    assert 0.0 <= prob <= 1.0

def test_predict_confidence_field():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=SAMPLE_STUDENT)
    assert r.json()["confidence"] in ["High", "Medium"]

def test_predict_missing_field_returns_422():
    r = client.post("/predict", json={"CGPA": 8.0})
    assert r.status_code == 422

def test_predict_invalid_cgpa_above_10():
    bad = {**SAMPLE_STUDENT, "CGPA": 15.0}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_batch_predict():
    _mock_model.predict.return_value       = np.array([1, 0])
    _mock_model.predict_proba.return_value = np.array([[0.1,0.9],[0.8,0.2]])
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict/batch", json={"students": [SAMPLE_STUDENT, SAMPLE_STUDENT]})
    assert r.status_code == 200
    assert r.json()["count"] == 2

def test_model_info_endpoint():
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", unittest_mock_open(_mock_meta)):
        r = client.get("/model/info")
    assert r.status_code == 200

def unittest_mock_open(data):
    import json
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(data))
