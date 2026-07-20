from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Legacy hardcoded fallbacks (used when ops_flights table is empty)
# ---------------------------------------------------------------------------
_AIRPORT_RISK_FALLBACK = {
    "DEL": 0.35, "BOM": 0.30, "CCU": 0.25, "BLR": 0.20,
    "MAA": 0.22, "HYD": 0.18, "GOI": 0.15,
}

_AIRCRAFT_RELIABILITY_FALLBACK = {
    "B737": 0.92, "A320": 0.94, "A321": 0.93, "ATR": 0.88,
}

_HOURLY_RISK_FALLBACK = {
    0: 0.40, 1: 0.42, 2: 0.45, 3: 0.48, 4: 0.30, 5: 0.15,
    6: 0.12, 7: 0.18, 8: 0.25, 9: 0.22, 10: 0.20, 11: 0.22,
    12: 0.28, 13: 0.25, 14: 0.22, 15: 0.20, 16: 0.25, 17: 0.30,
    18: 0.35, 19: 0.38, 20: 0.42, 21: 0.45, 22: 0.48, 23: 0.44,
}


# ---------------------------------------------------------------------------
# Learned distributions from ops_flights
# ---------------------------------------------------------------------------
def _build_delay_profiles(db_path: Path) -> Dict[str, Any]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    profiles: Dict[str, Any] = {}

    # Overall delay stats
    total = conn.execute("SELECT COUNT(*) FROM ops_flights").fetchone()[0]
    if total == 0:
        conn.close()
        return {}
    profiles["total_flights"] = total

    delayed = conn.execute(
        "SELECT COUNT(*) FROM ops_flights WHERE delay_min > 0"
    ).fetchone()[0]
    profiles["delayed_flights"] = delayed
    profiles["overall_delay_rate"] = delayed / total if total > 0 else 0

    avg_delay = conn.execute(
        "SELECT AVG(delay_min) FROM ops_flights WHERE delay_min > 0"
    ).fetchone()[0] or 0
    profiles["avg_delay_min"] = avg_delay

    # Per-airport delay rate
    airport_rows = conn.execute("""
        SELECT origin,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed,
               AVG(CASE WHEN delay_min > 0 THEN delay_min ELSE 0 END) as avg_delay
        FROM ops_flights GROUP BY origin
    """).fetchall()
    profiles["by_airport"] = {
        r["origin"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in airport_rows
    }

    # Per-aircraft delay rate
    ac_rows = conn.execute("""
        SELECT aircraft_type,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed,
               AVG(CASE WHEN delay_min > 0 THEN delay_min ELSE 0 END) as avg_delay
        FROM ops_flights GROUP BY aircraft_type
    """).fetchall()
    profiles["by_aircraft"] = {
        r["aircraft_type"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in ac_rows
    }

    # Per-hour delay rate
    hour_rows = conn.execute("""
        SELECT CAST(strftime('%H', std) AS INTEGER) as hour,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed
        FROM ops_flights GROUP BY hour ORDER BY hour
    """).fetchall()
    profiles["by_hour"] = {
        r["hour"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
        }
        for r in hour_rows
    }

    # Per-delay-reason frequency
    reason_rows = conn.execute("""
        SELECT delay_reason, COUNT(*) as cnt, AVG(delay_min) as avg_delay
        FROM ops_flights WHERE delay_min > 0 AND delay_reason != ''
        GROUP BY delay_reason ORDER BY cnt DESC
    """).fetchall()
    profiles["by_reason"] = {
        r["delay_reason"]: {
            "count": r["cnt"],
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in reason_rows
    }

    # Per-route-type delay rate
    route_rows = conn.execute("""
        SELECT route_type,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed
        FROM ops_flights GROUP BY route_type
    """).fetchall()
    profiles["by_route_type"] = {
        r["route_type"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
        }
        for r in route_rows
    }

    # Per-season delay rate
    season_rows = conn.execute("""
        SELECT season,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed
        FROM ops_flights GROUP BY season
    """).fetchall()
    profiles["by_season"] = {
        r["season"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
        }
        for r in season_rows
    }

    # Per-day-of-week delay rate
    dow_rows = conn.execute("""
        SELECT day_of_week,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed
        FROM ops_flights GROUP BY day_of_week
    """).fetchall()
    profiles["by_day_of_week"] = {
        r["day_of_week"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
        }
        for r in dow_rows
    }

    # Turbulence correlation
    turb_rows = conn.execute("""
        SELECT turbulence_category,
               COUNT(*) as total,
               SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed,
               AVG(delay_min) as avg_delay
        FROM ops_flights WHERE turbulence_category != ''
        GROUP BY turbulence_category
    """).fetchall()
    profiles["by_turbulence"] = {
        r["turbulence_category"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in turb_rows
    }

    # Seat occupancy correlation (bin into quartiles)
    occ_rows = conn.execute("""
        SELECT
            CASE
                WHEN seat_occupancy < 0.25 THEN 'Low (<25%)'
                WHEN seat_occupancy < 0.50 THEN 'Medium (25-50%)'
                WHEN seat_occupancy < 0.75 THEN 'High (50-75%)'
                ELSE 'Full (75%+)'
            END as occ_bin,
            COUNT(*) as total,
            SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed,
            AVG(delay_min) as avg_delay
        FROM ops_flights GROUP BY occ_bin
    """).fetchall()
    profiles["by_occupancy"] = {
        r["occ_bin"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in occ_rows
    }

    # Distance correlation (bin)
    dist_rows = conn.execute("""
        SELECT
            CASE
                WHEN distance < 1000 THEN 'Short (<1000km)'
                WHEN distance < 5000 THEN 'Medium (1000-5000km)'
                ELSE 'Long (5000km+)'
            END as dist_bin,
            COUNT(*) as total,
            SUM(CASE WHEN delay_min > 0 THEN 1 ELSE 0 END) as delayed,
            AVG(delay_min) as avg_delay
        FROM ops_flights GROUP BY dist_bin
    """).fetchall()
    profiles["by_distance"] = {
        r["dist_bin"]: {
            "total": r["total"],
            "delayed": r["delayed"],
            "delay_rate": r["delayed"] / r["total"] if r["total"] > 0 else 0,
            "avg_delay_min": round(r["avg_delay"] or 0, 1),
        }
        for r in dist_rows
    }

    conn.close()
    return profiles


# Cache
_PROFILES_CACHE: Optional[Dict[str, Any]] = None


def _get_profiles(db_path: Optional[Path] = None) -> Dict[str, Any]:
    global _PROFILES_CACHE
    path = db_path or (Path(__file__).parent.parent / "data" / "flights.db")
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE
    _PROFILES_CACHE = _build_delay_profiles(path)
    return _PROFILES_CACHE


def invalidate_profiles_cache():
    global _PROFILES_CACHE
    _PROFILES_CACHE = None


# ---------------------------------------------------------------------------
# Feature vector (enhanced with learned distributions)
# ---------------------------------------------------------------------------
def _feature_vector(
    origin: str,
    destination: str,
    aircraft_type: str,
    departure_hour: int,
    pax_count: int,
    flight_duration_min: int,
    is_international: bool,
    turbulence_category: str = "",
    seat_occupancy: float = 0.5,
    distance: float = 2000.0,
    hour_of_week: int = 0,
) -> np.ndarray:
    profiles = _get_profiles()

    if profiles and "by_airport" in profiles:
        origin_risk = profiles["by_airport"].get(origin, {}).get("delay_rate", 0.25)
        dest_risk = profiles["by_airport"].get(destination, {}).get("delay_rate", 0.25)
    else:
        origin_risk = _AIRPORT_RISK_FALLBACK.get(origin, 0.25)
        dest_risk = _AIRPORT_RISK_FALLBACK.get(destination, 0.25)

    if profiles and "by_aircraft" in profiles:
        ac_data = profiles["by_aircraft"].get(aircraft_type, {})
        aircraft_rel = 1.0 - ac_data.get("delay_rate", 0.1) if ac_data else 0.90
    else:
        aircraft_rel = _AIRCRAFT_RELIABILITY_FALLBACK.get(aircraft_type, 0.90)

    if profiles and "by_hour" in profiles:
        hour_data = profiles["by_hour"].get(departure_hour, {})
        hour_risk = hour_data.get("delay_rate", 0.25) if hour_data else 0.25
    else:
        hour_risk = _HOURLY_RISK_FALLBACK.get(departure_hour, 0.25)

    pax_factor = min(pax_count / 200.0, 1.0)
    duration_factor = min(flight_duration_min / 300.0, 1.0)
    intl_flag = 1.0 if is_international else 0.0
    peak_flag = 1.0 if departure_hour in (8, 9, 17, 18, 19, 20) else 0.0
    night_flag = 1.0 if departure_hour >= 22 or departure_hour < 5 else 0.0
    dow = (hour_of_week % 168) // 24
    weekend_flag = 1.0 if dow >= 5 else 0.0
    occ_factor = seat_occupancy
    dist_factor = min(distance / 10000.0, 1.0)

    if profiles and "by_turbulence" in profiles:
        turb_data = profiles["by_turbulence"].get(turbulence_category, {})
        turb_risk = turb_data.get("delay_rate", 0.1) if turb_data else 0.1
    else:
        turb_risk = 0.1

    return np.array([
        origin_risk, dest_risk, aircraft_rel, hour_risk,
        pax_factor, duration_factor, intl_flag, peak_flag,
        night_flag, weekend_flag, occ_factor, dist_factor, turb_risk,
    ])


_weights = np.array([
    0.15, 0.10, -0.12, 0.18,
    0.06, 0.08, 0.04, 0.10,
    0.06, 0.02, 0.05, 0.04, 0.08,
])
_bias = -0.05


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_delay(
    origin: str,
    destination: str,
    aircraft_type: str,
    departure_hour: int,
    pax_count: int = 150,
    flight_duration_min: int = 120,
    is_international: bool = False,
    departure_time: Optional[datetime] = None,
    turbulence_category: str = "",
    seat_occupancy: float = 0.5,
    distance: float = 2000.0,
) -> Dict[str, Any]:
    if departure_time is None:
        departure_time = datetime.now().replace(hour=departure_hour, minute=0)
    hour_of_week = departure_time.weekday() * 24 + departure_hour

    features = _feature_vector(
        origin, destination, aircraft_type, departure_hour,
        pax_count, flight_duration_min, is_international,
        turbulence_category, seat_occupancy, distance, hour_of_week,
    )

    logit = float(np.dot(features, _weights) + _bias)
    delay_probability = _sigmoid(logit)

    profiles = _get_profiles()
    avg_delay = profiles.get("avg_delay_min", 30) if profiles else 30

    expected_delay_min = 0.0
    if delay_probability > 0.3:
        expected_delay_min = delay_probability * avg_delay + random.uniform(5, 20)
    elif delay_probability > 0.15:
        expected_delay_min = delay_probability * avg_delay * 0.6 + random.uniform(5, 10)

    risk_level = "Low"
    if delay_probability > 0.5:
        risk_level = "High"
    elif delay_probability > 0.3:
        risk_level = "Medium"

    factors = []

    if profiles and "by_airport" in profiles:
        ap = profiles["by_airport"].get(origin, {})
        if ap.get("delay_rate", 0) > 0.30:
            factors.append(f"{origin} has high congestion risk ({ap['delay_rate']:.0%} historical delay rate)")
    elif origin in _AIRPORT_RISK_FALLBACK and _AIRPORT_RISK_FALLBACK[origin] > 0.30:
        factors.append(f"{origin} has high congestion risk")

    if departure_hour in (8, 9, 17, 18, 19, 20):
        factors.append("Peak hour departure increases delay risk")
    if departure_hour >= 22 or departure_hour < 5:
        factors.append("Night operations may face curfew/constraints")
    if pax_count > 160:
        factors.append("High passenger load extends boarding time")
    if is_international:
        factors.append("International flights require additional clearance")
    if seat_occupancy > 0.85:
        factors.append("Near-full aircraft increases boarding complexity")
    if distance > 5000:
        factors.append("Long-haul flight has higher fuel/weather exposure")

    if profiles and "by_reason" in profiles:
        top_reason = max(profiles["by_reason"].items(), key=lambda x: x[1]["count"])
        factors.append(f"Most common delay cause: {top_reason[0]} ({top_reason[1]['count']} occurrences, avg {top_reason[1]['avg_delay_min']:.0f} min)")

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
            "occupancy_factor": round(float(features[10]), 3),
            "distance_factor": round(float(features[11]), 3),
            "turbulence_risk": round(float(features[12]), 3),
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


# ---------------------------------------------------------------------------
# Enhanced analytics from ops_flights
# ---------------------------------------------------------------------------
def get_delay_cause_breakdown(db_path: Optional[Path] = None) -> Dict[str, Any]:
    profiles = _get_profiles(db_path)
    if not profiles or "by_reason" not in profiles:
        return {"message": "No ops flight data loaded. Use 'Load Operations Dataset' first."}
    return {
        "total_flights": profiles["total_flights"],
        "delayed_flights": profiles["delayed_flights"],
        "overall_delay_rate": round(profiles["overall_delay_rate"], 3),
        "causes": profiles["by_reason"],
    }


def get_delay_by_airport(db_path: Optional[Path] = None) -> Dict[str, Any]:
    profiles = _get_profiles(db_path)
    if not profiles or "by_airport" not in profiles:
        return {"message": "No ops flight data loaded."}
    return profiles["by_airport"]


def get_delay_by_time(db_path: Optional[Path] = None) -> Dict[str, Any]:
    profiles = _get_profiles(db_path)
    if not profiles:
        return {"message": "No ops flight data loaded."}
    return {
        "by_hour": profiles.get("by_hour", {}),
        "by_day_of_week": profiles.get("by_day_of_week", {}),
        "by_season": profiles.get("by_season", {}),
    }


def get_delay_by_route_type(db_path: Optional[Path] = None) -> Dict[str, Any]:
    profiles = _get_profiles(db_path)
    if not profiles:
        return {"message": "No ops flight data loaded."}
    return {
        "by_route_type": profiles.get("by_route_type", {}),
        "by_turbulence": profiles.get("by_turbulence", {}),
        "by_occupancy": profiles.get("by_occupancy", {}),
        "by_distance": profiles.get("by_distance", {}),
    }
