from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.opensky_db import get_feature_table, get_filtered_feature_table, DEFAULT_DB_PATH


FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "is_peak_hour",
    "route_encoded",
    "airport_congestion",
    "prev_flight_delay",
    "aircraft_daily_flights",
    "wind_speed_knots",
    "wind_gust_knots",
    "visibility_m",
    "cloud_cover_pct",
    "precipitation_mm",
    "temperature_c",
    "pressure_hpa",
    "weather_severity",
    "wind_severity",
    "has_rain",
    "has_high_wind",
    "has_low_visibility",
    "wind_cloud_interaction",
    "wind_precip_interaction",
]

TARGET_BINARY = "is_delayed"
TARGET_REGRESSION = "deviation_min"

_PEAK_HOURS = {8, 9, 17, 18, 19, 20}


def _compute_route_encoded(df: pd.DataFrame) -> pd.Series:
    routes = df["origin_airport"].fillna("") + "_" + df["destination_airport"].fillna("")
    unique_routes = sorted(routes.unique())
    route_map = {r: i for i, r in enumerate(unique_routes)}
    return routes.map(route_map).fillna(-1).astype(int)


def _compute_route_labels(df: pd.DataFrame) -> pd.Series:
    return (df["origin_airport"].fillna("") + "_" + df["destination_airport"].fillna(""))


def _compute_congestion(df: pd.DataFrame) -> pd.Series:
    if "first_seen" not in df.columns:
        return pd.Series(0, index=df.index)
    ts = pd.to_datetime(df["first_seen"], unit="s", utc=True)
    hour_key = ts.dt.floor("h")
    counts = hour_key.groupby(hour_key).transform("count")
    return counts.fillna(0).astype(int)


def _compute_aircraft_daily_flights(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns or "icao24" not in df.columns:
        return pd.Series(1, index=df.index)
    group_key = df["icao24"].fillna("") + "_" + df["date"].fillna("")
    counts = group_key.groupby(group_key).transform("count")
    return counts.fillna(1).astype(int)


def _compute_prev_flight_delay(df: pd.DataFrame) -> pd.Series:
    if "prev_flight_delay_min" in df.columns:
        return df["prev_flight_delay_min"].fillna(0)
    return pd.Series(0.0, index=df.index)


def _compute_wind_knots(kmh: pd.Series) -> pd.Series:
    return kmh.fillna(0) * 0.539957


def _compute_weather_severity(features: pd.DataFrame) -> pd.Series:
    severity = pd.Series(0.0, index=features.index)
    severity += (features["wind_speed_knots"] / 30).clip(0, 1) * 0.3
    severity += (features["precipitation_mm"] / 10).clip(0, 1) * 0.3
    severity += ((10000 - features["visibility_m"]).clip(0, 10000) / 10000) * 0.2
    severity += (features["cloud_cover_pct"] / 100) * 0.2
    return severity


def _compute_wind_severity(features: pd.DataFrame) -> pd.Series:
    wind = features["wind_speed_knots"]
    severity = pd.Series(0.0, index=features.index)
    severity = severity.where(wind <= 15, 1.0)
    severity = severity.where(wind <= 25, 2.0)
    severity = severity.where(wind <= 35, 3.0)
    severity = severity.where(wind <= 45, 4.0)
    return severity


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    features = pd.DataFrame(index=df.index)

    if "first_seen" in df.columns:
        dt = pd.to_datetime(df["first_seen"], unit="s", utc=True)
        features["hour_of_day"] = dt.dt.hour
        features["day_of_week"] = dt.dt.dayofweek
        features["month"] = dt.dt.month
    else:
        features["hour_of_day"] = 12
        features["day_of_week"] = 0
        features["month"] = 1

    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)
    features["is_peak_hour"] = features["hour_of_day"].isin(_PEAK_HOURS).astype(int)

    features["route_encoded"] = _compute_route_encoded(df)

    features["airport_congestion"] = _compute_congestion(df)
    features["prev_flight_delay"] = _compute_prev_flight_delay(df)
    features["aircraft_daily_flights"] = _compute_aircraft_daily_flights(df)

    wind_kmh = df.get("wind_speed_kmh", pd.Series(0, index=df.index))
    features["wind_speed_knots"] = _compute_wind_knots(wind_kmh)

    gust_kmh = df.get("wind_gusts_kmh", pd.Series(0, index=df.index))
    features["wind_gust_knots"] = _compute_wind_knots(gust_kmh)

    features["visibility_m"] = df.get("visibility_m", pd.Series(10000, index=df.index)).fillna(10000)
    features["cloud_cover_pct"] = df.get("cloud_cover_pct", pd.Series(0, index=df.index)).fillna(0)
    features["precipitation_mm"] = df.get("precipitation_mm", pd.Series(0, index=df.index)).fillna(0)
    features["temperature_c"] = df.get("temperature_c", pd.Series(25, index=df.index)).fillna(25)
    features["pressure_hpa"] = df.get("pressure_hpa", pd.Series(1013, index=df.index)).fillna(1013)

    features["weather_severity"] = _compute_weather_severity(features)
    features["wind_severity"] = _compute_wind_severity(features)
    features["has_rain"] = (features["precipitation_mm"] > 0.5).astype(int)
    features["has_high_wind"] = (features["wind_speed_knots"] > 20).astype(int)
    features["has_low_visibility"] = (features["visibility_m"] < 5000).astype(int)
    features["wind_cloud_interaction"] = features["wind_speed_knots"] * features["cloud_cover_pct"] / 100
    features["wind_precip_interaction"] = features["wind_speed_knots"] * features["precipitation_mm"]

    for col in FEATURE_COLUMNS:
        if col not in features.columns:
            features[col] = 0
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)

    return features[FEATURE_COLUMNS]


def get_training_data(
    min_samples: int = 30,
    callsigns: Optional[List[str]] = None,
    db_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if callsigns:
        raw = get_filtered_feature_table(callsigns, db_path)
    else:
        raw = get_feature_table(db_path)
    min_samples = 5 if callsigns else 30
    if raw.empty or len(raw) < min_samples:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=str)

    same_origin = raw["origin_airport"].fillna("") == raw["destination_airport"].fillna("")
    raw = raw[~same_origin].copy()

    if len(raw) < min_samples:
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=str)

    route_labels = _compute_route_labels(raw)
    features = build_features(raw)

    binary_target = raw[TARGET_BINARY].fillna(0).astype(int) if TARGET_BINARY in raw.columns else pd.Series(0, index=raw.index)
    reg_target = raw[TARGET_REGRESSION].fillna(0) if TARGET_REGRESSION in raw.columns else pd.Series(0.0, index=raw.index)

    valid_mask = features.notna().all(axis=1)
    features = features[valid_mask]
    binary_target = binary_target[valid_mask]
    reg_target = reg_target[valid_mask]
    route_labels = route_labels[valid_mask]

    return features, binary_target, reg_target, route_labels


def build_single_flight_features(
    hour_of_day: int,
    day_of_week: int,
    month: int,
    route: str,
    prev_delay: float = 0.0,
    airport_congestion: int = 0,
    aircraft_flights_today: int = 1,
    wind_speed_kmh: float = 0.0,
    wind_gusts_kmh: float = 0.0,
    visibility_m: float = 10000.0,
    cloud_cover_pct: float = 0.0,
    precipitation_mm: float = 0.0,
    temperature_c: float = 25.0,
    pressure_hpa: float = 1013.0,
) -> pd.DataFrame:
    wind_knots = wind_speed_kmh * 0.539957
    weather_sev = min(1.0, (wind_knots / 30) * 0.3 + (precipitation_mm / 10) * 0.3 + max(0, (10000 - visibility_m) / 10000) * 0.2 + (cloud_cover_pct / 100) * 0.2)
    wind_sev = 0.0 if wind_knots <= 15 else (1.0 if wind_knots <= 25 else (2.0 if wind_knots <= 35 else (3.0 if wind_knots <= 45 else 4.0)))

    data = {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": int(day_of_week >= 5),
        "is_peak_hour": int(hour_of_day in _PEAK_HOURS),
        "route_encoded": hash(route) % 1000,
        "airport_congestion": airport_congestion,
        "prev_flight_delay": prev_delay,
        "aircraft_daily_flights": aircraft_flights_today,
        "wind_speed_knots": wind_knots,
        "wind_gust_knots": wind_gusts_kmh * 0.539957,
        "visibility_m": visibility_m,
        "cloud_cover_pct": cloud_cover_pct,
        "precipitation_mm": precipitation_mm,
        "temperature_c": temperature_c,
        "pressure_hpa": pressure_hpa,
        "weather_severity": weather_sev,
        "wind_severity": wind_sev,
        "has_rain": int(precipitation_mm > 0.5),
        "has_high_wind": int(wind_knots > 20),
        "has_low_visibility": int(visibility_m < 5000),
        "wind_cloud_interaction": wind_knots * cloud_cover_pct / 100,
        "wind_precip_interaction": wind_knots * precipitation_mm,
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)
