"""Aplikasi prediksi churn nasabah bank (FastAPI + Vue.js).

Jalankan:  uvicorn app:app --reload
Buka:      http://127.0.0.1:8000
"""
import json
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model_def import FEATURES, NUMERIC, ChurnNet

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model" / "churn_model.pth"
META_PATH = BASE / "model" / "preprocess.json"

app = FastAPI(title="Prediksi Churn Nasabah Bank")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

model = None
meta = None
load_error = None


def load_model():
    """Memuat model .pth dan parameter preprocessing dari folder model/."""
    global model, meta, load_error
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        net = ChurnNet()
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        net.eval()
        model = net
        load_error = None
    except Exception as e:  # file belum ada / tidak cocok
        model, meta = None, None
        load_error = f"Model belum bisa dimuat: {e}"


load_model()


class Nasabah(BaseModel):
    CreditScore: float = Field(ge=300, le=900)
    Geography: str
    Gender: str
    Age: float = Field(ge=18, le=100)
    Tenure: float = Field(ge=0, le=10)
    Balance: float = Field(ge=0)
    NumOfProducts: int = Field(ge=1, le=4)
    HasCrCard: int = Field(ge=0, le=1)
    IsActiveMember: int = Field(ge=0, le=1)
    EstimatedSalary: float = Field(ge=0)


def to_vector(d: Nasabah):
    if d.Geography not in ("France", "Germany", "Spain"):
        raise HTTPException(422, "Geography harus France, Germany, atau Spain")
    if d.Gender not in ("Male", "Female"):
        raise HTTPException(422, "Gender harus Male atau Female")
    row = {
        "CreditScore": d.CreditScore, "Age": d.Age, "Tenure": d.Tenure, "Balance": d.Balance,
        "NumOfProducts": d.NumOfProducts, "HasCrCard": d.HasCrCard,
        "IsActiveMember": d.IsActiveMember, "EstimatedSalary": d.EstimatedSalary,
        "Geography_France": int(d.Geography == "France"),
        "Geography_Germany": int(d.Geography == "Germany"),
        "Geography_Spain": int(d.Geography == "Spain"),
        "Gender_Male": int(d.Gender == "Male"),
    }
    for c in NUMERIC:  # standardisasi dengan rata-rata dan std dari data latih
        row[c] = (row[c] - meta["mean"][c]) / meta["std"][c]
    return torch.tensor([[row[f] for f in FEATURES]], dtype=torch.float32)


def risk_level(p: float):
    """Pita risiko tetap berdasarkan peluang churn (rata-rata churn di data sekitar 20%, model menandai nasabah mulai sekitar 9%)."""
    if p >= 0.50:
        return "tinggi", "Risiko tinggi. Hubungi nasabah, tawarkan bunga khusus atau layanan prioritas."
    if p >= 0.15:
        return "sedang", "Risiko sedang. Pantau dan tawarkan produk yang sesuai."
    return "rendah", "Risiko rendah. Pertahankan layanan seperti biasa."


@app.get("/")
def index():
    return FileResponse(BASE / "templates" / "index.html")


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "error": load_error}


@app.get("/model-info")
def model_info():
    if meta is None:
        return JSONResponse({"loaded": False, "error": load_error}, status_code=503)
    return {"loaded": True, "metrics": meta.get("metrics", {}), "threshold": meta.get("threshold", 0.5)}


@app.post("/predict")
def predict(d: Nasabah):
    if model is None:
        raise HTTPException(503, load_error or "Model belum dimuat")
    x = to_vector(d)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).item()
    thr = meta.get("threshold", 0.5)
    level, advice = risk_level(p)
    return {
        "churn_probability": round(p, 4),
        "will_churn": p >= thr,
        "risk_level": level,
        "advice": advice,
    }
