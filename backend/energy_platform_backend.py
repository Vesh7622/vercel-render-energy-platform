from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.services import get_health, get_overview, get_forecast

app = FastAPI(title="Energy Platform Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return get_health()

@app.get("/api/overview")
def overview():
    return get_overview()

@app.get("/api/forecast")
def forecast():
    return get_forecast()

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "backend/models/hybrid_load_forecaster.h5.ipynb"
SCALER_X_PATH = MODEL_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"