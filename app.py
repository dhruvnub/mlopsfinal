# app.py
# Experiment 4: FastAPI-based ML Inference API
# Offline: uvicorn app:app --reload  → http://127.0.0.1:8000
# Online:  Deployed to Azure App Service via GitHub Actions

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import joblib, json, os

app = FastAPI(
    title="Placement Prediction API",
    description="ML inference API for student placement prediction. Deployed via Azure App Service.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — required for ui.html to call this API from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Safe lazy model loader ─────────────────────────────────────────────────
_model    = None
_metadata = {}

def load_model():
    global _model, _metadata
    if _model is None:
        if not os.path.exists("models/model.pkl"):
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Run: python train.py"
            )
        _model = joblib.load("models/model.pkl")
        if os.path.exists("models/metadata.json"):
            with open("models/metadata.json") as f:
                _metadata = json.load(f)
    return _model

# ── Schemas ────────────────────────────────────────────────────────────────
class Student(BaseModel):
    CGPA:              float = Field(..., ge=0, le=10,  example=8.5)
    Internships:       int   = Field(..., ge=0, le=10,  example=2)
    Projects:          int   = Field(..., ge=0, le=20,  example=3)
    AptitudeTestScore: int   = Field(..., ge=0, le=100, example=80)
    SoftSkillsRating:  float = Field(..., ge=0, le=5,   example=4.2)
    SSC_Marks:         int   = Field(..., ge=0, le=100, example=78)
    HSC_Marks:         int   = Field(..., ge=0, le=100, example=75)

class BatchRequest(BaseModel):
    students: list[Student]

# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def serve_ui():
    if os.path.exists("ui.html"):
        return FileResponse("ui.html")
    return {"message": "Placement Prediction API", "docs": "/docs"}

@app.get("/health", tags=["System"])
def health():
    return {
        "status":      "healthy",
        "model_ready": os.path.exists("models/model.pkl"),
        "version":     "1.0.0"
    }

@app.get("/model/info", tags=["Model"])
def model_info():
    if not os.path.exists("models/metadata.json"):
        return {"status": "not_trained", "message": "Run train.py first"}
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return {"status": "loaded", **meta}

@app.post("/predict", tags=["Inference"])
def predict(student: Student):
    model = load_model()
    X = [[
        student.CGPA, student.Internships, student.Projects,
        student.AptitudeTestScore, student.SoftSkillsRating,
        student.SSC_Marks, student.HSC_Marks,
    ]]
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    return {
        "placement_status":   "Placed" if pred == 1 else "Not Placed",
        "probability_placed": round(float(proba), 4),
        "confidence":         "High" if proba > 0.75 or proba < 0.25 else "Medium",
        "model_type":         _metadata.get("model_type", ""),
        "run_id":             _metadata.get("run_id", ""),
    }

@app.post("/predict/batch", tags=["Inference"])
def predict_batch(batch: BatchRequest):
    model = load_model()
    results = []
    for s in batch.students:
        X = [[s.CGPA, s.Internships, s.Projects,
              s.AptitudeTestScore, s.SoftSkillsRating,
              s.SSC_Marks, s.HSC_Marks]]
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0][1]
        results.append({
            "placement_status":   "Placed" if pred == 1 else "Not Placed",
            "probability_placed": round(float(proba), 4),
        })
    return {"count": len(results), "predictions": results}
