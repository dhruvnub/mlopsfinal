# tests/test_app.py
# Experiment 2: Unit tests — run with: pytest tests/ -v
# Fully mocked so tests pass BEFORE training (CI safe)

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Mock model that returns sensible values
_mock_model = MagicMock()
_mock_model.predict.return_value       = np.array([1])
_mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])

_mock_meta = {
    "run_id":     "ci-test-run-001",
    "model_type": "LogisticRegression",
    "features":   ["CGPA", "Internships", "Projects",
                   "AptitudeTestScore", "SoftSkillsRating",
                   "SSC_Marks", "HSC_Marks"],
    "metrics":    {
        "accuracy": 0.7985, "f1": 0.7571,
        "precision": 0.7659, "recall": 0.7485, "roc_auc": 0.8675
    }
}

# Patch before importing app
import app as app_module
app_module._model    = _mock_model
app_module._metadata = _mock_meta

from app import app
client = TestClient(app)

GOOD_STUDENT = {
    "CGPA": 8.5, "Internships": 2, "Projects": 3,
    "AptitudeTestScore": 82, "SoftSkillsRating": 4.2,
    "SSC_Marks": 80, "HSC_Marks": 76
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"

def test_predict_status():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=GOOD_STUDENT)
    assert r.status_code == 200
    assert r.json()["placement_status"] in ["Placed", "Not Placed"]

def test_predict_probability_range():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=GOOD_STUDENT)
    assert 0.0 <= r.json()["probability_placed"] <= 1.0

def test_predict_confidence_field():
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict", json=GOOD_STUDENT)
    assert r.json()["confidence"] in ["High", "Medium"]

def test_predict_missing_field_returns_422():
    r = client.post("/predict", json={"CGPA": 8.0})
    assert r.status_code == 422

def test_predict_cgpa_above_10_returns_422():
    bad = {**GOOD_STUDENT, "CGPA": 15.0}
    r   = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_batch_predict():
    _mock_model.predict.return_value       = np.array([1, 0])
    _mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])
    with patch("app._model", _mock_model), patch("app._metadata", _mock_meta):
        r = client.post("/predict/batch", json={"students": [GOOD_STUDENT, GOOD_STUDENT]})
    assert r.status_code == 200
    assert r.json()["count"] == 2
