from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

ENERGY_DB_PATH = Path(os.getenv("ENERGY_DB_PATH", DATA_DIR / "energy_platform.db"))
ENERGY_LOOKBACK = int(os.getenv("ENERGY_LOOKBACK", "24"))
MODEL_PATH = MODELS_DIR / "gru_lstm_model.keras"
SCALER_X_PATH = MODELS_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODELS_DIR / "scaler_y.pkl"
