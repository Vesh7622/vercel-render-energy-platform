from __future__ import annotations

import pandas as pd
import requests

LAT = 52.3676
LON = 4.9041


def fetch_live_weather() -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&hourly=temperature_2m,direct_radiation,diffuse_radiation,"
        "relative_humidity_2m,cloud_cover,wind_speed_10m"
        "&forecast_days=2"
        "&timezone=UTC"
    )

    res = requests.get(url, timeout=30)
    res.raise_for_status()

    data = res.json()
    hourly = data.get("hourly", {})

    df = pd.DataFrame({
        "observed_at": hourly.get("time", []),
        "temperature_2m": hourly.get("temperature_2m", []),
        "direct_radiation": hourly.get("direct_radiation", []),
        "diffuse_radiation": hourly.get("diffuse_radiation", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "cloud_cover": hourly.get("cloud_cover", []),
        "wind_speed_10m": hourly.get("wind_speed_10m", []),
    })

    if df.empty:
        return df

    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["wind_speed_80m"] = df["wind_speed_10m"]
    df["shortwave_radiation"] = df["direct_radiation"] + df["diffuse_radiation"]
    df["sunshine_duration"] = None
    df["source"] = "open-meteo"

    return df