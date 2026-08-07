from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from ml_engine import delay_predictor as dp


def _seed_prediction_log(db_path, rows=None, prob=0.1, expected_min=30.0):
    if rows is None:
        rows = [(1, 80.0)] * 20
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        for i, (delayed, minutes) in enumerate(rows):
            conn.execute(
                """INSERT OR REPLACE INTO prediction_log
                   (callsign, origin, destination, date, predicted_at,
                    delay_probability, expected_delay_min, risk_level, predicted_delayed,
                    actual_is_delayed, actual_delay_min)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"VT{i:03d}", "BLR", "DEL", today, now_iso,
                    prob, expected_min, "Low", 0, delayed, minutes,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _correction_row(db_path, route="BLR_DEL"):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT logit_bias, minutes_bias, sample_count FROM prediction_correction WHERE route = ?",
            (route,),
        ).fetchone()
    finally:
        conn.close()


def test_update_online_correction_adjusts_drift(db_path):
    _seed_prediction_log(db_path)
    result = dp.update_online_correction(db_path=db_path)
    assert result["status"] == "success"
    assert result["routes_with_actuals"] >= 1
    assert result["routes_adjusted"] >= 1

    row = _correction_row(db_path)
    assert row is not None
    assert row[0] > 0  # positive logit bias toward delayed
    assert row[1] > 0  # positive minutes bias
    assert row[2] == 20


def test_update_online_correction_no_drift_no_adjustment(db_path):
    rows = [(1 if i % 2 == 0 else 0, 15.0) for i in range(20)]
    _seed_prediction_log(db_path, rows=rows, prob=0.5, expected_min=15.0)
    result = dp.update_online_correction(db_path=db_path)
    assert result["status"] == "success"
    assert result["routes_adjusted"] == 0


def test_reset_online_corrections_clears_table(db_path):
    _seed_prediction_log(db_path)
    dp.update_online_correction(db_path=db_path)
    assert _correction_row(db_path) is not None
    dp._reset_online_corrections(db_path)
    assert _correction_row(db_path) is None


def test_online_correction_feeds_predict_delay(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(dp, "_load_metadata", lambda: {})
    from data.flights_db import init_db
    from data.opensky_db import init_opensky_tables

    path = init_db(db_path=tmp_path / "flights.db")
    init_opensky_tables(path)
    _seed_prediction_log(path)
    dp.update_online_correction(db_path=path)

    result = dp.predict_delay(
        origin="BLR", destination="DEL", departure_hour=12, db_path=path
    )
    assert result["delay_probability"] > 0.0
    assert "delay_probability" in result
