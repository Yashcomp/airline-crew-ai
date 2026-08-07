from __future__ import annotations

import sqlite3

from data import opensky_db


def test_tables_exist(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "prediction_correction" in tables
    assert "model_registry" in tables
    assert "prediction_log" in tables
    assert "delay_labels" in tables


def test_register_and_history(db_path):
    metadata = {
        "trained_at": "2026-08-07T09:45:00+00:00",
        "samples": 100,
        "train_samples": 80,
        "test_samples": 20,
        "delay_rate": 0.3,
        "n_routes": 10,
        "cv_accuracy": 0.9,
        "cv_f1": 0.5,
        "cv_roc_auc": 0.8,
        "cv_mae": 5.0,
        "cv_rmse": 7.0,
        "optimal_threshold": {"threshold": 0.11},
        "blend_weight": 0.7,
        "feature_importance": {"wind": 0.1},
        "message": "test run",
    }
    res = opensky_db.register_model_run("v20260807_0001", metadata, {}, db_path=db_path)
    assert res["status"] == "registered"

    history = opensky_db.get_model_history(db_path=db_path)
    assert len(history) >= 1
    row = history.iloc[0]
    assert row["version"] == "v20260807_0001"
    assert row["status"] == "active"
    assert row["optimal_threshold"] == 0.11


def test_register_archives_previous_active(db_path):
    opensky_db.register_model_run(
        "v1", {"trained_at": "2026-08-01T00:00:00+00:00"}, {}, db_path=db_path
    )
    opensky_db.register_model_run(
        "v2", {"trained_at": "2026-08-02T00:00:00+00:00"}, {}, db_path=db_path
    )
    history = opensky_db.get_model_history(db_path=db_path)
    statuses = dict(zip(history["version"], history["status"]))
    assert statuses["v1"] == "archived"
    assert statuses["v2"] == "active"


def test_activate_model_version_round_trip(monkeypatch, tmp_path, db_path):
    monkeypatch.setattr(opensky_db, "_MODEL_ARTIFACT_DIR", tmp_path)
    snapshot = tmp_path / "delay_classifier_v1.pkl"
    snapshot.write_bytes(b"model-bytes")

    opensky_db.register_model_run(
        "v1",
        {"trained_at": "2026-08-07T00:00:00+00:00"},
        {"delay_classifier.pkl": "delay_classifier_v1.pkl"},
        db_path=db_path,
    )
    res = opensky_db.activate_model_version("v1", db_path=db_path)
    assert res["status"] == "activated"
    assert (tmp_path / "delay_classifier.pkl").read_bytes() == b"model-bytes"

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT status FROM model_registry WHERE version = 'v1'").fetchone()
    finally:
        conn.close()
    assert row[0] == "active"


def test_log_predictions_and_update_actuals(db_path):
    today = "2026-08-07"
    flight = {
        "callsign": "VT999",
        "origin": "BLR",
        "destination": "DEL",
        "scheduled_date": today,
        "prediction": {
            "delay_probability": 0.8,
            "expected_delay_min": 45.0,
            "risk_level": "High",
            "predicted_delayed": True,
        },
    }
    assert opensky_db.log_predictions([flight], db_path=db_path, target_date=today) == 1

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """INSERT INTO delay_labels
               (flight_id, origin, destination, date, departure_hour, day_of_week,
                actual_duration_min, expected_duration_min, deviation_min, is_delayed)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("VT999", "BLR", "DEL", today, 12, 4, 150, 120, 30, 1),
        )
        conn.commit()
    finally:
        conn.close()

    updated = opensky_db.update_actuals(db_path=db_path)
    assert updated >= 1

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT actual_is_delayed, actual_delay_min FROM prediction_log WHERE callsign = 'VT999'"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1
    assert row[1] == 30
