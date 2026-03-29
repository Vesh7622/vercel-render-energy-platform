from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import ENERGY_LOOKBACK

FEATURE_COLUMNS = [
    "total_generation_mw",
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_speed_80m",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "hour_of_day",
    "day_of_week",
    "month_of_year",
]

def build_model_input(feature_df: pd.DataFrame) -> Optional[np.ndarray]:
    if feature_df.empty or len(feature_df) < ENERGY_LOOKBACK:
        return None
    working = feature_df[FEATURE_COLUMNS].copy().ffill().bfill()
    values = working.values.astype("float32")
    return np.expand_dims(values, axis=0)

class PersistenceFallbackModel:
    def predict(self, model_input: np.ndarray, verbose: int = 0) -> np.ndarray:
        _ = verbose
        last_generation = float(model_input[0, -1, 0])
        return np.array([[last_generation]], dtype=np.float32)
