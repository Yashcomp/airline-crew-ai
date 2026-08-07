from __future__ import annotations

import math

import pytest

from ml_engine import delay_predictor as dp


def test_apply_correction_passthrough_without_bias(monkeypatch):
    monkeypatch.setattr(dp, "_load_route_correction", lambda route, db_path=None: {})
    prob, minutes = dp.apply_online_correction(0.4, 25.0, "BLR_DEL", db_path="x")
    assert prob == pytest.approx(0.4)
    assert minutes == pytest.approx(25.0)


def test_apply_correction_applies_logit_bias(monkeypatch):
    monkeypatch.setattr(
        dp, "_load_route_correction",
        lambda route, db_path=None: {"logit_bias": 0.5, "minutes_bias": 0.0},
    )
    prob, minutes = dp.apply_online_correction(0.4, 25.0, "BLR_DEL", db_path="x")
    assert prob > 0.4
    assert minutes == pytest.approx(25.0)


def test_apply_correction_applies_minutes_bias(monkeypatch):
    monkeypatch.setattr(
        dp, "_load_route_correction",
        lambda route, db_path=None: {"logit_bias": 0.0, "minutes_bias": 15.0},
    )
    prob, minutes = dp.apply_online_correction(0.4, 25.0, "BLR_DEL", db_path="x")
    assert prob == pytest.approx(0.4)
    assert minutes == pytest.approx(40.0)


def test_apply_correction_clamps_probability(monkeypatch):
    monkeypatch.setattr(
        dp, "_load_route_correction",
        lambda route, db_path=None: {"logit_bias": -20.0, "minutes_bias": 0.0},
    )
    prob, minutes = dp.apply_online_correction(0.9, 60.0, "BLR_DEL", db_path="x")
    assert 0.05 <= prob <= 0.95


def test_threshold_fallback_without_metadata(monkeypatch):
    monkeypatch.setattr(dp, "_load_metadata", lambda: {})
    assert dp.get_prediction_threshold() == pytest.approx(0.30)


def test_threshold_from_metadata(monkeypatch):
    monkeypatch.setattr(
        dp, "_load_metadata",
        lambda: {"optimal_threshold": {"threshold": 0.11}},
    )
    assert dp.get_prediction_threshold() == pytest.approx(0.11)


def test_predict_delay_heuristic_structure(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "_MODELS_DIR", tmp_path)
    monkeypatch.setattr(dp, "_load_metadata", lambda: {})
    monkeypatch.setattr(dp, "_load_route_correction", lambda route, db_path=None: {})
    result = dp.predict_delay(
        origin="BLR", destination="DEL", aircraft_type="B737",
        departure_hour=12, db_path=tmp_path / "empty.db",
    )
    for key in ("delay_probability", "expected_delay_min", "risk_level", "predicted_delayed", "factors"):
        assert key in result
    assert 0.0 <= result["delay_probability"] <= 1.0
    assert math.isfinite(result["expected_delay_min"])
