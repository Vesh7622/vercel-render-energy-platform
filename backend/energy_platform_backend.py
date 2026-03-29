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

@app.get("/api/performance")
def api_performance():
    metric_row = latest_row("model_metrics", "created_at")
    training_row = latest_row("training_runs", "created_at")
    loss_df = read_sql(
        "SELECT epoch, train_loss, val_loss FROM training_loss_history ORDER BY epoch ASC"
    )
    pred_df = read_sql(
        "SELECT observed_at, actual_mw, predicted_mw FROM prediction_evaluations ORDER BY observed_at ASC LIMIT 200"
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


@app.get("/api/comparison")
def api_comparison():
    df = read_sql(
        "SELECT model_name, rmse, mae, mse, training_time_sec, selected FROM baseline_comparisons ORDER BY rmse ASC"
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
            "The deployed model was selected because it achieved the lowest forecasting error "
            "among the evaluated baseline approaches."
            if not df.empty else
            "No baseline comparison data available yet."
        ),
    }


@app.get("/api/xai")
def api_xai():
    global_df = read_sql(
        "SELECT feature_name, importance FROM xai_global_importance ORDER BY importance DESC"
    )
    local_df = read_sql(
        "SELECT feature_name, contribution FROM xai_local_explanations ORDER BY ABS(contribution) DESC"
    )

    return {
        "summary": (
            "Explainability output is generated from stored global importance values and local forecast contributions."
            if not global_df.empty or not local_df.empty else
            "No explainability data available yet."
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


@app.get("/api/forecast-history")
def api_forecast_history():
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


@app.get("/api/freshness")
def api_freshness():
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


@app.get("/api/data-status")
def api_data_status():
    generation_rows = read_sql(
        "SELECT observed_at, total_generation_mw, generation_wind_mw, generation_solar_mw FROM generation_observations ORDER BY observed_at DESC LIMIT 20"
    ).sort_values("observed_at")

    weather_rows = read_sql(
        "SELECT observed_at, temperature_2m, wind_speed_10m, shortwave_radiation FROM weather_observations ORDER BY observed_at DESC LIMIT 20"
    ).sort_values("observed_at")

    feature_rows = read_sql(
        '''
        SELECT observed_at, total_generation_mw, temperature_2m, wind_speed_80m, hour_of_day
        FROM model_features
        ORDER BY observed_at DESC
        LIMIT 20
        '''
    ).sort_values("observed_at")

    return {
        "generationRows": generation_rows.replace({np.nan: None}).to_dict(orient="records"),
        "weatherRows": weather_rows.replace({np.nan: None}).to_dict(orient="records"),
        "featureRows": feature_rows.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/system-status")
def api_system_status():
    return {
        "databasePath": str(DB_PATH),
        "modelPath": str(MODEL_PATH),
        "scalerXPath": str(SCALER_X_PATH),
        "scalerYPath": str(SCALER_Y_PATH),
        "modelLoaded": RUNTIME_ASSETS.get("model") is not None,
        "databaseCounts": {
            "weather_observations": table_count("weather_observations"),
            "generation_observations": table_count("generation_observations"),
            "model_features": table_count("model_features"),
            "forecasts": table_count("forecasts"),
            "scenario_runs": table_count("scenario_runs"),
        },
    }


@app.post("/api/scenario")
def api_scenario(payload: ScenarioPayload):
    feature_df = recent_features(DEFAULT_LOOKBACK)
    baseline = baseline_forecast_value(feature_df)
    scenario_df = apply_scenario(feature_df, payload)
    scenario_value = baseline_forecast_value(scenario_df)
    delta = scenario_value - baseline

    return {
        "baselineMw": round(float(baseline), 2),
        "scenarioMw": round(float(scenario_value), 2),
        "deltaMw": round(float(delta), 2),
        "explanation": (
            "The scenario result was generated by modifying the latest weather-driven inputs "
            "and rerunning the live forecasting pipeline."
        ),
    }


conn.executescript(
    """
    CREATE TABLE IF NOT EXISTS training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT,
        completed_at TEXT,
        epochs INTEGER,
        batch_size INTEGER,
        notes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS model_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_label TEXT,
        rmse REAL,
        mae REAL,
        mse REAL,
        mape REAL,
        accuracy REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS training_loss_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        epoch INTEGER NOT NULL,
        train_loss REAL,
        val_loss REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS prediction_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        observed_at TEXT NOT NULL,
        actual_mw REAL,
        predicted_mw REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS baseline_comparisons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_name TEXT NOT NULL UNIQUE,
        rmse REAL,
        mae REAL,
        mse REAL,
        training_time_sec REAL,
        selected INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS xai_global_importance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL UNIQUE,
        importance REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS xai_local_explanations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feature_name TEXT NOT NULL UNIQUE,
        contribution REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS forecast_archive (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        issued_at TEXT NOT NULL,
        forecast_for TEXT NOT NULL,
        predicted_mw REAL,
        actual_mw REAL,
        absolute_error REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS data_source_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        last_success TEXT,
        latency_minutes REAL,
        next_expected_run TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """
)

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "backend/models/hybrid_load_forecaster.h5.ipynb"
SCALER_X_PATH = MODEL_DIR / "scaler_x.pkl"
SCALER_Y_PATH = MODEL_DIR / "scaler_y.pkl"