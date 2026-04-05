from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import requests


URL = "https://transparency.entsoe.eu/load/total/dayAhead/load"

HEADERS = {
    "accept": "application/json",
    "content-type": "application/json; charset=utf-8",
    "origin": "https://transparency.entsoe.eu",
    "referer": "https://transparency.entsoe.eu/load/total/dayAhead",
    "user-agent": "Mozilla/5.0",
}


def build_payload(days_back: int = 3) -> dict:
    now = datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days_back)

    return {
        "dateTimeRange": {
            "from": start.isoformat().replace("+00:00", ".000Z"),
            "to": end.isoformat().replace("+00:00", ".000Z"),
        },
        "areaList": ["BZN|10YNL----------L"],
        "filterMap": {},
        "sorterList": [],
        "timeZone": "CET",
    }


def fetch_load_data() -> pd.DataFrame:
    payload = build_payload()

    response = requests.post(URL, json=payload, headers=HEADERS, timeout=60)
    response.raise_for_status()

    data = response.json()

    instance = data["instanceList"][0]
    period = instance["curveData"]["periodList"][0]
    point_map = period["pointMap"]
    start_time = datetime.fromisoformat(
        period["timeInterval"]["from"].replace("Z", "+00:00")
    )

    rows = []

    for idx_str, values in point_map.items():
        idx = int(idx_str)
        ts = start_time + timedelta(minutes=15 * idx)

        # For the load page, these usually map to:
        # values[0] = day-ahead total load forecast
        # values[1] = actual total load
        forecast_val = values[0] if len(values) > 0 else None
        actual_val = values[1] if len(values) > 1 else None

        rows.append(
            {
                "observed_at": ts,
                "load_forecast_mw": float(forecast_val) if isinstance(forecast_val, str) else None,
                "load_actual_mw": float(actual_val) if isinstance(actual_val, str) else None,
                "source": "entsoe-load-hidden-endpoint",
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("observed_at").reset_index(drop=True)