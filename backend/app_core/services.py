import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app_core.config import (
    DB_PATH,
    MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    ENERGY_LOOKBACK,
)
from app_core.db import initialize_schema, read_sql, latest_row, table_count, get_connection
from app_core.schemas import ScenarioPayload

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


MODEL = None
FEATURE_SCALER = None
TARGET_SCALER = None

FEATURE_COLUMNS = [
    "load_actual_mw",
    "load_forecast_mw",
    "temperature_c",
    "rad_direct",
    "rad_diffuse",
    "hour",
    "minute",
    "dayofweek",
    "month",
    "is_weekend",
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "lag_15m",
    "lag_30m",
    "lag_45m",
    "lag_60m",
    "temperature_lag_15m",
    "temperature_lag_60m",
    "rad_direct_lag_15m",
    "rad_direct_lag_60m",
    "rad_diffuse_lag_15m",
    "rad_diffuse_lag_60m",
    "roll_mean_past_1h",
    "roll_std_past_1h",
]


def init_backend() -> None:
    global MODEL, FEATURE_SCALER, TARGET_SCALER

    initialize_schema()

    if MODEL_PATH.exists() and load_model is not None:
        MODEL = load_model(MODEL_PATH, compile=False)

    if SCALER_X_PATH.exists():
        FEATURE_SCALER = joblib.load(SCALER_X_PATH)

    if SCALER_Y_PATH.exists():
        TARGET_SCALER = joblib.load(SCALER_Y_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_ready() -> bool:
    return MODEL is not None and FEATURE_SCALER is not None and TARGET_SCALER is not None


def _build_engineered_features(base_df: pd.DataFrame) -> pd.DataFrame:
    if base_df.empty:
        return pd.DataFrame(columns=["observed_at"] + FEATURE_COLUMNS)

    df = base_df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["observed_at"]).sort_values("observed_at").drop_duplicates("observed_at")

    for col in ["total_generation_mw", "temperature_2m", "direct_radiation", "diffuse_radiation"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.set_index("observed_at")
    df = df.resample("15min").interpolate(method="time").ffill().bfill()

    df["load_actual_mw"] = df["total_generation_mw"]
    df["load_forecast_mw"] = df["load_actual_mw"]
    df["temperature_c"] = df["temperature_2m"]
    df["rad_direct"] = df["direct_radiation"]
    df["rad_diffuse"] = df["diffuse_radiation"]

    minutes_of_day = df.index.hour * 60 + df.index.minute

    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["sin_hour"] = np.sin(2 * np.pi * minutes_of_day / 1440.0)
    df["cos_hour"] = np.cos(2 * np.pi * minutes_of_day / 1440.0)
    df["sin_dow"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dow"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    df["lag_15m"] = df["load_actual_mw"].shift(1)
    df["lag_30m"] = df["load_actual_mw"].shift(2)
    df["lag_45m"] = df["load_actual_mw"].shift(3)
    df["lag_60m"] = df["load_actual_mw"].shift(4)

    df["temperature_lag_15m"] = df["temperature_c"].shift(1)
    df["temperature_lag_60m"] = df["temperature_c"].shift(4)

    df["rad_direct_lag_15m"] = df["rad_direct"].shift(1)
    df["rad_direct_lag_60m"] = df["rad_direct"].shift(4)

    df["rad_diffuse_lag_15m"] = df["rad_diffuse"].shift(1)
    df["rad_diffuse_lag_60m"] = df["rad_diffuse"].shift(4)

    df["roll_mean_past_1h"] = df["load_actual_mw"].rolling(window=4, min_periods=1).mean()
    df["roll_std_past_1h"] = df["load_actual_mw"].rolling(window=4, min_periods=1).std().fillna(0.0)

    df = df.ffill().bfill()

    return df.reset_index()[["observed_at"] + FEATURE_COLUMNS]


def _prepare_feature_window() -> pd.DataFrame:
    base_df = read_sql(
        """
        SELECT observed_at, total_generation_mw, temperature_2m, direct_radiation, diffuse_radiation
        FROM model_features
        ORDER BY observed_at DESC
        LIMIT 96
        """
    ).sort_values("observed_at")

    engineered = _build_engineered_features(base_df)
    return engineered.tail(ENERGY_LOOKBACK)


def _predict_from_features(feature_df: pd.DataFrame) -> float:
    if feature_df.empty or len(feature_df) < ENERGY_LOOKBACK:
        return 0.0

    if not _model_ready():
        return float(feature_df.iloc[-1]["load_actual_mw"])

    x = feature_df[FEATURE_COLUMNS].copy()
    x = x.ffill().bfill()

    scaled_x = FEATURE_SCALER.transform(x)
    model_input = np.expand_dims(scaled_x.astype("float32"), axis=0)

    expected_shape = getattr(MODEL, "input_shape", None)
    if expected_shape is not None:
        expected_timesteps = expected_shape[1]
        expected_features = expected_shape[2]
        if model_input.shape[1] != expected_timesteps or model_input.shape[2] != expected_features:
            raise ValueError(
                f"Model expects shape (None, {expected_timesteps}, {expected_features}) "
                f"but received {model_input.shape}"
            )

    pred_scaled = MODEL.predict(model_input, verbose=0)
    pred = TARGET_SCALER.inverse_transform(np.array(pred_scaled).reshape(-1, 1))[0][0]

    return float(pred)


def get_health() -> dict:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        database_connected = True
    except Exception:
        database_connected = False

    if database_connected and _model_ready():
        status = "ok"
    elif database_connected:
        status = "degraded"
    else:
        status = "error"

    return {
        "status": status,
        "databaseConnected": database_connected,
        "modelLoaded": _model_ready(),
        "latestRefresh": _now_iso(),
    }


def get_overview() -> dict:
    actual_df = read_sql(
        "SELECT observed_at, total_generation_mw FROM generation_observations ORDER BY observed_at DESC LIMIT 24"
    ).sort_values("observed_at")

    forecast_df = read_sql(
        "SELECT forecast_for, forecast_generation_mw FROM forecasts ORDER BY forecast_for ASC LIMIT 12"
    )

    weather_df = read_sql(
        "SELECT observed_at, temperature_2m, wind_speed_10m, shortwave_radiation FROM weather_observations ORDER BY observed_at DESC LIMIT 24"
    ).sort_values("observed_at")

    latest_generation = float(actual_df.iloc[-1]["total_generation_mw"]) if not actual_df.empty else 0.0
    latest_temperature = float(weather_df.iloc[-1]["temperature_2m"]) if not weather_df.empty else 0.0

    return {
        "latestGenerationMw": round(latest_generation, 2),
        "latestTemperatureC": round(latest_temperature, 2),
        "lookbackWindow": ENERGY_LOOKBACK,
        "storedForecasts": table_count("forecasts"),
        "actualSeries": [
            {
                "time": pd.to_datetime(row["observed_at"]).strftime("%H:%M"),
                "generation": round(float(row["total_generation_mw"]), 2),
            }
            for _, row in actual_df.iterrows()
        ],
        "forecastSeries": [
            {
                "time": pd.to_datetime(row["forecast_for"]).strftime("%H:%M"),
                "forecast": round(float(row["forecast_generation_mw"]), 2),
            }
            for _, row in forecast_df.iterrows()
        ],
        "weatherSeries": [
            {
                "time": pd.to_datetime(row["observed_at"]).strftime("%H:%M"),
                "temperature": round(float(row["temperature_2m"]), 2),
                "windSpeed": round(float(row["wind_speed_10m"]), 2),
                "radiation": round(float(row["shortwave_radiation"]), 2),
            }
            for _, row in weather_df.iterrows()
        ],
    }


def get_forecast() -> dict:
    feature_df = _prepare_feature_window()
    next_forecast = _predict_from_features(feature_df)

    return {
        "nextForecastMw": round(next_forecast, 2),
        "generatedAt": _now_iso(),
        "horizonHours": 1,
        "recentInputs": feature_df.tail(8).where(pd.notnull(feature_df), None).to_dict(orient="records"),
    }


def get_performance() -> dict:
    metric_row = latest_row("model_metrics", "created_at")
    training_row = latest_row("training_runs", "created_at")
    loss_df = read_sql(
        """
        SELECT epoch, train_loss, val_loss
        FROM training_loss_history
        ORDER BY epoch ASC
        """
    )
    pred_df = read_sql(
        """
        SELECT observed_at, actual_mw, predicted_mw
        FROM prediction_evaluations
        ORDER BY observed_at ASC
        LIMIT 300
        """
    )

    return {
        "metrics": {
            "rmse": round(float(metric_row["rmse"]), 2) if metric_row else 0.0,
            "mae": round(float(metric_row["mae"]), 2) if metric_row else 0.0,
            "mse": round(float(metric_row["mse"]), 2) if metric_row else 0.0,
            "mape": round(float(metric_row["mape"]), 2) if metric_row else 0.0,
            "accuracy": round(float(metric_row["accuracy"]), 2) if metric_row else 0.0,
        },
        "trainingInfo": {
            "lastTrainingAt": training_row["completed_at"] if training_row else None,
            "lastRetrainingAt": training_row["completed_at"] if training_row else None,
            "epochs": int(training_row["epochs"] or 0) if training_row else 0,
            "batchSize": int(training_row["batch_size"] or 0) if training_row else 0,
        },
        "lossCurve": [
            {
                "epoch": int(row["epoch"]),
                "trainLoss": float(row["train_loss"]) if row["train_loss"] is not None else 0.0,
                "valLoss": float(row["val_loss"]) if row["val_loss"] is not None else 0.0,
            }
            for _, row in loss_df.iterrows()
        ],
        "actualVsPredicted": [
            {
                "time": pd.to_datetime(row["observed_at"]).strftime("%m-%d %H:%M"),
                "actual": round(float(row["actual_mw"]), 2),
                "predicted": round(float(row["predicted_mw"]), 2),
            }
            for _, row in pred_df.iterrows()
        ],
    }


def get_comparison() -> dict:
    df = read_sql(
        """
        SELECT model_name, rmse, mae, mse, training_time_sec, selected
        FROM baseline_comparisons
        ORDER BY rmse ASC
        """
    )

    return {
        "models": [
            {
                "name": str(row["model_name"]),
                "rmse": round(float(row["rmse"]), 2),
                "mae": round(float(row["mae"]), 2),
                "mse": round(float(row["mse"]), 2),
                "trainingTimeSec": round(float(row["training_time_sec"]), 2),
                "selected": bool(int(row["selected"])),
            }
            for _, row in df.iterrows()
        ],
        "summary": (
            "The deployed model was selected because it achieved the lowest forecasting error among the evaluated baseline approaches."
            if not df.empty
            else "No baseline comparison data available yet."
        ),
    }


def get_xai() -> dict:
    global_df = read_sql(
        "SELECT feature_name, importance FROM xai_global_importance ORDER BY importance DESC"
    )
    local_df = read_sql(
        "SELECT feature_name, contribution FROM xai_local_explanations ORDER BY ABS(contribution) DESC"
    )

    return {
        "summary": (
            "Explainability output is generated from stored global importance values and local forecast contributions."
            if not global_df.empty or not local_df.empty
            else "No explainability data available yet."
        ),
        "featureImportance": [
            {"feature": str(row["feature_name"]), "importance": round(float(row["importance"]), 4)}
            for _, row in global_df.iterrows()
        ],
        "localExplanation": [
            {"feature": str(row["feature_name"]), "contribution": round(float(row["contribution"]), 2)}
            for _, row in local_df.iterrows()
        ],
    }


def get_forecast_history() -> dict:
    df = read_sql(
        "SELECT issued_at, forecast_for, predicted_mw, actual_mw, absolute_error FROM forecast_archive ORDER BY forecast_for ASC LIMIT 200"
    )

    return {
        "rows": [
            {
                "issuedAt": row["issued_at"],
                "forecastFor": row["forecast_for"],
                "predictedMw": round(float(row["predicted_mw"]), 2),
                "actualMw": round(float(row["actual_mw"]), 2),
                "absoluteError": round(float(row["absolute_error"]), 2),
            }
            for _, row in df.iterrows()
        ]
    }


def get_freshness() -> dict:
    df = read_sql(
        "SELECT source_name, status, last_success, latency_minutes, next_expected_run FROM data_source_status ORDER BY source_name ASC"
    )

    return {
        "sources": [
            {
                "name": str(row["source_name"]),
                "status": str(row["status"]),
                "lastSuccess": row["last_success"],
                "latencyMinutes": int(float(row["latency_minutes"] or 0)),
                "nextExpectedRun": row["next_expected_run"],
            }
            for _, row in df.iterrows()
        ]
    }


def get_data_status() -> dict:
    generation_rows = read_sql(
        "SELECT observed_at, total_generation_mw, generation_wind_mw, generation_solar_mw FROM generation_observations ORDER BY observed_at DESC LIMIT 20"
    ).sort_values("observed_at")

    weather_rows = read_sql(
        "SELECT observed_at, temperature_2m, wind_speed_10m, shortwave_radiation FROM weather_observations ORDER BY observed_at DESC LIMIT 20"
    ).sort_values("observed_at")

    feature_rows = read_sql(
        """
        SELECT observed_at, total_generation_mw, temperature_2m, wind_speed_80m, hour_of_day
        FROM model_features
        ORDER BY observed_at DESC
        LIMIT 20
        """
    ).sort_values("observed_at")

    return {
        "generationRows": generation_rows.where(pd.notnull(generation_rows), None).to_dict(orient="records"),
        "weatherRows": weather_rows.where(pd.notnull(weather_rows), None).to_dict(orient="records"),
        "featureRows": feature_rows.where(pd.notnull(feature_rows), None).to_dict(orient="records"),
    }


def get_system_status() -> dict:
    return {
        "databasePath": str(DB_PATH),
        "modelPath": str(MODEL_PATH),
        "scalerXPath": str(SCALER_X_PATH),
        "scalerYPath": str(SCALER_Y_PATH),
        "modelLoaded": _model_ready(),
        "databaseCounts": {
            "weather_observations": table_count("weather_observations"),
            "generation_observations": table_count("generation_observations"),
            "model_features": table_count("model_features"),
            "forecasts": table_count("forecasts"),
            "scenario_runs": table_count("scenario_runs"),
        },
    }


def run_scenario(payload: ScenarioPayload) -> dict:
    feature_df = _prepare_feature_window()
    baseline = _predict_from_features(feature_df)

    if feature_df.empty:
        scenario_value = baseline
    else:
        modified = feature_df.copy()
        idx = modified.index[-1]

        modified.loc[idx, "temperature_c"] = float(modified.loc[idx, "temperature_c"]) + payload.temperatureDelta
        modified.loc[idx, "rad_direct"] = float(modified.loc[idx, "rad_direct"]) * payload.radiationMultiplier
        modified.loc[idx, "rad_diffuse"] = float(modified.loc[idx, "rad_diffuse"]) * payload.radiationMultiplier
        modified.loc[idx, "load_forecast_mw"] = float(modified.loc[idx, "load_forecast_mw"]) * payload.windMultiplier

        scenario_value = _predict_from_features(modified)

    delta = scenario_value - baseline

    explanation = (
        "The scenario result was generated by modifying the latest weather-driven inputs "
        "and rerunning the live forecasting model."
    )

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO scenario_runs (
                scenario_name, generated_at, changed_variables,
                baseline_generation_mw, scenario_generation_mw, delta_mw, explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.scenarioName,
                _now_iso(),
                json.dumps(payload.model_dump()),
                float(baseline),
                float(scenario_value),
                float(delta),
                explanation,
            ),
        )
        conn.commit()

    return {
        "baselineMw": round(float(baseline), 2),
        "scenarioMw": round(float(scenario_value), 2),
        "deltaMw": round(float(delta), 2),
        "explanation": explanation,
    }