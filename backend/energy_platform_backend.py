from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from app_core.services import clean_json
from app_core.entsoe_client import fetch_load_data
from app_core.schemas import ScenarioPayload
from app_core.services import (
    init_backend,
    get_health,
    get_overview,
    get_forecast,
    get_performance,
    get_comparison,
    get_xai,
    get_forecast_history,
    get_freshness,
    get_data_status,
    get_system_status,
    run_scenario,
)

init_backend()

app = FastAPI(title="Energy Platform Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Energy Platform Backend is running"}


@app.get("/api/health")
def health():
    return get_health()


@app.get("/api/overview")
def overview():
    return get_overview()


@app.get("/api/forecast")
def forecast():
    return get_forecast()


@app.get("/api/performance")
def performance():
    return get_performance()


@app.get("/api/comparison")
def comparison():
    return get_comparison()


@app.get("/api/xai")
def xai():
    return get_xai()


@app.get("/api/forecast-history")
def forecast_history():
    return get_forecast_history()


@app.get("/api/freshness")
def freshness():
    return get_freshness()


@app.get("/api/data-status")
def data_status():
    return get_data_status()


@app.get("/api/system-status")
def system_status():
    return get_system_status()


@app.post("/api/scenario")
def scenario(payload: ScenarioPayload):
    return run_scenario(payload)


@app.get("/api/live-load")
def api_live_load():
    df = fetch_load_data()
    records = df.tail(20).to_dict(orient="records")
    return clean_json(records)