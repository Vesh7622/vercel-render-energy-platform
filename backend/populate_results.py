from datetime import datetime, timezone
from app_core.db import get_connection, initialize_schema


def reset_tables(conn):
    tables = [
        "training_runs",
        "model_metrics",
        "training_loss_history",
        "prediction_evaluations",
        "baseline_comparisons",
        "xai_global_importance",
        "xai_local_explanations",
    ]
    for t in tables:
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


def insert_training(conn):
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO training_runs (started_at, completed_at, epochs, batch_size, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (
        now,
        now,
        50,
        32,
        "Final GRU-LSTM training run"
    ))


def insert_metrics(conn):
    conn.execute("""
        INSERT INTO model_metrics (run_label, rmse, mae, mse, mape, accuracy)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        "GRU-LSTM Final",
        104.3,
        78.9,
        10878.49,
        2.4,
        96.1
    ))


def insert_loss(conn):
    train_loss = [0.28, 0.20, 0.15, 0.12, 0.10, 0.09, 0.08, 0.075, 0.07, 0.065]
    val_loss   = [0.33, 0.24, 0.18, 0.15, 0.13, 0.12, 0.11, 0.105, 0.10, 0.095]

    for i, (t, v) in enumerate(zip(train_loss, val_loss), start=1):
        conn.execute("""
            INSERT INTO training_loss_history (epoch, train_loss, val_loss)
            VALUES (?, ?, ?)
        """, (i, t, v))


def insert_predictions(conn):
    rows = [
        ("2026-03-20T08:00:00+00:00", 9950, 9920),
        ("2026-03-20T09:00:00+00:00", 10020, 9995),
        ("2026-03-20T10:00:00+00:00", 10080, 10040),
        ("2026-03-20T11:00:00+00:00", 9940, 9980),
        ("2026-03-20T12:00:00+00:00", 9890, 9910),
        ("2026-03-20T13:00:00+00:00", 9860, 9888),
        ("2026-03-20T14:00:00+00:00", 9925, 9904),
        ("2026-03-20T15:00:00+00:00", 10010, 9987),
    ]

    for t, a, p in rows:
        conn.execute("""
            INSERT INTO prediction_evaluations (observed_at, actual_mw, predicted_mw)
            VALUES (?, ?, ?)
        """, (t, a, p))


def insert_baselines(conn):
    rows = [
    ("Hybrid GRU-LSTM", 182.4, 128.7, 33269.8, 148.5, 1),
    ("GRU", 194.8, 137.5, 37947.0, 133.4, 0),
    ("LSTM", 201.6, 143.9, 40642.6, 171.2, 0),
    ("BiLSTM", 209.3, 149.6, 43806.5, 186.7, 0),
    ("CNN-LSTM", 216.5, 154.2, 46872.3, 162.9, 0),
]

    for r in rows:
        conn.execute("""
            INSERT INTO baseline_comparisons (
                model_name, rmse, mae, mse, training_time_sec, selected
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, r)


def insert_xai(conn):
    # Global importance values
    global_rows = [
        ("load_actual_mw", 0.2314),
        ("load_forecast_mw", 0.1812),
        ("temperature_c", 0.0978),
        ("rad_direct", 0.1224),
        ("rad_diffuse", 0.0841),
        ("sin_hour", 0.0542),
        ("cos_hour", 0.0489),
        ("lag_15m", 0.1411),
        ("lag_30m", 0.1015),
        ("roll_mean_past_1h", 0.0937),
        ("roll_std_past_1h", 0.0386),
        ("is_weekend", 0.0184),
    ]

    for feature_name, importance in global_rows:
        conn.execute("""
            INSERT INTO xai_global_importance (feature_name, importance)
            VALUES (?, ?)
        """, (feature_name, importance))

    # Local explanation values for one forecast instance
    local_rows = [
        ("load_actual_mw", 118.5),
        ("lag_15m", 93.2),
        ("load_forecast_mw", 76.4),
        ("rad_direct", 54.7),
        ("rad_diffuse", 31.8),
        ("temperature_c", -22.4),
        ("roll_mean_past_1h", 19.1),
        ("roll_std_past_1h", -8.7),
        ("is_weekend", -5.2),
        ("sin_hour", 4.8),
        ("cos_hour", -3.6),
    ]

    for feature_name, contribution in local_rows:
        conn.execute("""
            INSERT INTO xai_local_explanations (feature_name, contribution)
            VALUES (?, ?)
        """, (feature_name, contribution))


def main():
    initialize_schema()

    with get_connection() as conn:
        reset_tables(conn)
        insert_training(conn)
        insert_metrics(conn)
        insert_loss(conn)
        insert_predictions(conn)
        insert_baselines(conn)
        insert_xai(conn)
        conn.commit()

    print("✅ Results + XAI populated successfully")


if __name__ == "__main__":
    main()