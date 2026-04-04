# train.py
# Experiment 6: Track experiments using MLflow → Azure ML integration
# Experiment 8: Evaluate and register models → Azure ML Model Registry
#
# Offline:  python train.py                          (logs to local mlruns/)
# Online:   MLFLOW_TRACKING_URI=<azure-uri> python train.py

import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score
)
import mlflow
import mlflow.sklearn
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# MLflow Setup — reads MLFLOW_TRACKING_URI from environment
# Offline : defaults to local "mlruns" folder
# Online  : set MLFLOW_TRACKING_URI to your Azure ML tracking URI
#           azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/...
# ─────────────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
EXPERIMENT_NAME = "placement-prediction"
MODEL_NAME      = "placement-classifier"   # name in Azure ML Model Registry

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ─────────────────────────────────────────────────────────────────────────────
# Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/placementdata.csv")

FEATURES = [
    "CGPA", "Internships", "Projects",
    "AptitudeTestScore", "SoftSkillsRating",
    "SSC_Marks", "HSC_Marks"
]

X = df[FEATURES]
y = (df["PlacementStatus"] == "Placed").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────────────────────────────────────
# Train two models, log both, register the best one
# ─────────────────────────────────────────────────────────────────────────────
candidates = {
    "RandomForest": {
        "model": RandomForestClassifier(
            n_estimators=150, max_depth=8,
            min_samples_split=5, random_state=42
        ),
        "params": {"n_estimators": 150, "max_depth": 8, "min_samples_split": 5}
    },
    "LogisticRegression": {
        "model": LogisticRegression(C=1.0, max_iter=500, random_state=42),
        "params": {"C": 1.0, "max_iter": 500}
    }
}

best_f1    = 0
best_run   = None
best_model = None
best_name  = None
best_meta  = {}

for model_name, cfg in candidates.items():
    with mlflow.start_run(run_name=f"{model_name}-run") as run:
        clf = cfg["model"]
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, preds),  4),
            "f1":        round(f1_score(y_test, preds),        4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall":    round(recall_score(y_test, preds),    4),
            "roc_auc":   round(roc_auc_score(y_test, proba),   4),
        }
        cv_f1 = cross_val_score(clf, X, y, cv=5, scoring="f1").mean()

        # Log everything to MLflow (Experiment 6)
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("dataset",    "placement-10k")
        mlflow.log_params(cfg["params"])
        mlflow.log_params({"model_type": model_name, "test_size": 0.2})
        mlflow.log_metrics(metrics)
        mlflow.log_metric("cv_f1", round(cv_f1, 4))

        # Log model to MLflow / Azure ML Model Registry (Experiment 8)
        mlflow.sklearn.log_model(
            clf,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print(f"\n  [{model_name}]")
        for k, v in metrics.items():
            print(f"    {k:12s}: {v}")

        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_run   = run.info.run_id
            best_model = clf
            best_name  = model_name
            best_meta  = {"run_id": best_run, "model_type": best_name,
                          "features": FEATURES, "metrics": metrics}

# ─────────────────────────────────────────────────────────────────────────────
# Save best model locally for the FastAPI (Experiment 4)
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
with open("models/metadata.json", "w") as f:
    json.dump(best_meta, f, indent=2)

print(f"\n{'='*55}")
print(f"  Best Model : {best_name}")
print(f"  F1 Score   : {best_f1}")
print(f"  Run ID     : {best_run}")
print(f"  Saved      : models/model.pkl + models/metadata.json")
print(f"{'='*55}")
print(f"\n  View runs  : mlflow ui  → http://127.0.0.1:5000")
print(f"  Azure ML   : https://ml.azure.com\n")
