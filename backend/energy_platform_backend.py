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
