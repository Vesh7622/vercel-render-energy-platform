from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DB_PATH = Path(os.getenv("ENERGY_DB_PATH", DATA_DIR / "energy_platform.db"))
MODEL_PATH = Path(os.getenv("ENERGY_MODEL_PATH", MODEL_DIR / "hybrid_load_forecaster.h5"))
SCALER_X_PATH = Path(os.getenv("ENERGY_SCALER_X_PATH", MODEL_DIR / "feature_scaler.pkl"))
SCALER_Y_PATH = Path(os.getenv("ENERGY_SCALER_Y_PATH", MODEL_DIR / "target_scaler.pkl"))

ENERGY_LOOKBACK = int(os.getenv("ENERGY_LOOKBACK", "16"))

OPEN_METEO_BASE_URL = os.getenv("OPEN_METEO_BASE_URL", "https://api.open-meteo.com")
ENTSOE_API_TOKEN = os.getenv("ENTSOE_API_TOKEN", "")
