from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


_airport_risk = {
    "DEL": 0.35, "BOM": 0.30, "CCU": 0.25, "BLR": 0.20,
    "MAA": 0.22, "HYD": 0.18, "GOI": 0.15,
}

_aircraft_reliability = {
    "B737": 0.92, "A320": 0.94, "A321": 0.93, "ATR": 0.88,
}

_hourly_risk = {
    0: 0.40, 1: 0.42, 2: 0.45, 3: 0.48, 4: 0.30, 5: 0.15,
    6: 0.12, 7: 0.18, 8: 0.25, 9: 0.22, 10: 0.20, 11: 0.22,
    12: 0.28, 13: 0.25, 14: 0.22, 15: 0.20, 16: 0.25, 17: 0.30,
    18: 0.35, 19: 0.38, 20: 0.42, 21: 0.45, 22: 0.48, 23: 0.44,
}


def _feature_vector(
    origin: str,
    destination: str,
    aircraft_type: str,
    departure_hour: int,
    pax_count: int,
    flight_duration_min: int,
    is_international: bool,
    hour_of_week: int = 0,
) -> np.ndarray:
    origin_risk = _airport_risk.get(origin, 0.25)
    dest_risk = _airport_risk.get(destination, 0.25)
    aircraft_rel = _aircraft_reliability.get(aircraft_type, 0.90)
    hour_risk = _hourly_risk.get(departure_hour, 0.25)
    pax_factor = min(pax_count / 200.0, 1.0)
    duration_factor = min(flight_duration_min / 300.0, 1.0)
    intl_flag = 1.0 if is_international else 0.0
    peak_flag = 1.0 if departure_hour in (8, 9, 17, 18, 19, 20) else 0.0
    night_flag = 1.0 if departure_hour >= 22 or departure_hour < 5 else 0.0
    dow = (hour_of_week % 168) // 24
    weekend_flag = 1.0 if dow >= 5 else 0.0

    return np.array([
        origin_risk, dest_risk, aircraft_rel, hour_risk,
        pax_factor, duration_factor, intl_flag, peak_flag,
        night_flag, weekend_flag,
    ])


_weights = np.array([
    0.18, 0.12, -0.15, 0.22,
    0.08, 0.10, 0.05, 0.12,
    0.08, 0.03,
])
_bias = -0.05


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def predict_delay(
    origin: str,
    destination: str,
    aircraft_type: str,
    departure_hour: int,
    pax_count: int = 150,
    flight_duration_min: int = 120,
    is_international: bool = False,
    departure_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    if departure_time is None:
        departure_time = datetime.now().replace(hour=departure_hour, minute=0)
    hour_of_week = departure_time.weekday() * 24 + departure_hour

    features = _feature_vector(
        origin, destination, aircraft_type, departure_hour,
        pax_count, flight_duration_min, is_international, hour_of_week,
    )

    logit = float(np.dot(features, _weights) + _bias)
    delay_probability = _sigmoid(logit)

    expected_delay_min = 0.0
    if delay_probability > 0.3:
        expected_delay_min = delay_probability * 45 + random.uniform(5, 20)
    elif delay_probability > 0.15:
        expected_delay_min = delay_probability * 25 + random.uniform(5, 10)

    risk_level = "Low"
    if delay_probability > 0.5:
        risk_level = "High"
    elif delay_probability > 0.3:
        risk_level = "Medium"

    factors = []
    if origin_risk := _airport_risk.get(origin, 0):
        if origin_risk > 0.30:
            factors.append(f"{origin} has high congestion risk ({origin_risk:.0%})")
    if departure_hour in (8, 9, 17, 18, 19, 20):
        factors.append("Peak hour departure increases delay risk")
    if departure_hour >= 22 or departure_hour < 5:
        factors.append("Night operations may face curfew/constraints")
    if pax_count > 160:
        factors.append("High passenger load extends boarding time")
    if is_international:
        factors.append("International flights require additional clearance")

    return {
        "delay_probability": round(delay_probability, 3),
        "expected_delay_min": round(expected_delay_min, 1),
        "risk_level": risk_level,
        "factors": factors,
        "features": {
            "origin_risk": round(float(features[0]), 3),
            "dest_risk": round(float(features[1]), 3),
            "aircraft_reliability": round(float(features[2]), 3),
            "hour_risk": round(float(features[3]), 3),
        },
    }


def train_delay_model(flight_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(flight_data) < 5:
        return {
            "status": "insufficient_data",
            "message": "Need at least 5 flights to train. Using heuristic model.",
            "samples": len(flight_data),
        }

    correct = 0
    total = len(flight_data)
    for record in flight_data:
        predicted = predict_delay(
            origin=record.get("origin", "DEL"),
            destination=record.get("destination", "BOM"),
            aircraft_type=record.get("aircraft_type", "B737"),
            departure_hour=record.get("departure_hour", 12),
            pax_count=record.get("pax_count", 150),
            flight_duration_min=record.get("flight_duration_min", 120),
            is_international=record.get("is_international", False),
        )
        was_delayed = record.get("actual_delay_min", 0) > 15
        predicted_delayed = predicted["delay_probability"] > 0.3
        if was_delayed == predicted_delayed:
            correct += 1

    accuracy = correct / total
    return {
        "status": "trained",
        "accuracy": round(accuracy, 3),
        "samples": total,
        "correct": correct,
        "message": f"Model trained on {total} samples with {accuracy:.1%} accuracy.",
    }


def get_delay_insights(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.flights_db import get_flights
    flights = get_flights(db_path=db_path)
    if not flights:
        return {"message": "No flight data available for insights."}

    delayed = [f for f in flights if f.status.value in ("delayed", "cancelled")]
    total = len(flights)
    delay_rate = len(delayed) / total if total > 0 else 0

    by_origin: Dict[str, Dict[str, int]] = {}
    by_hour: Dict[int, int] = {}
    by_aircraft: Dict[str, int] = {}

    for f in delayed:
        by_origin.setdefault(f.origin, {"delayed": 0, "total": 0})
        by_origin[f.origin]["delayed"] += 1

        by_hour.setdefault(f.std.hour, 0)
        by_hour[f.std.hour] += 1

        by_aircraft.setdefault(f.aircraft_type, 0)
        by_aircraft[f.aircraft_type] += 1

    for f in flights:
        by_origin.setdefault(f.origin, {"delayed": 0, "total": 0})
        by_origin[f.origin]["total"] += 1

    risk_scores = {}
    for airport, counts in by_origin.items():
        rate = counts["delayed"] / counts["total"] if counts["total"] > 0 else 0
        risk_scores[airport] = round(rate, 3)

    peak_hours = sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_flights": total,
        "delayed_flights": len(delayed),
        "delay_rate": round(delay_rate, 3),
        "airport_risk_scores": risk_scores,
        "peak_delay_hours": [{"hour": h, "count": c} for h, c in peak_hours],
        "aircraft_delays": by_aircraft,
    }
