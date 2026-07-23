from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_BLR_LAT = 13.1986
_BLR_LON = 77.7066

_HOURLY_VARS = ",".join([
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "visibility",
    "cloud_cover",
    "cloud_cover_low",
    "precipitation",
    "pressure_msl",
    "weather_code",
])

DEFAULT_DB_PATH = Path(__file__).parent / "flights.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_weather_table(db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weather_cache (
            timestamp TEXT PRIMARY KEY,
            temperature_c REAL,
            wind_speed_kmh REAL,
            wind_direction_deg REAL,
            wind_gusts_kmh REAL,
            visibility_m REAL,
            cloud_cover_pct REAL,
            cloud_cover_low_pct REAL,
            precipitation_mm REAL,
            pressure_hpa REAL,
            weather_code INTEGER
        )
    """)
    conn.commit()
    conn.close()


def get_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = _BLR_LAT,
    longitude: float = _BLR_LON,
) -> List[Dict[str, Any]]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": _HOURLY_VARS,
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    result = []
    for i, t in enumerate(times):
        result.append({
            "timestamp": t,
            "temperature_c": hourly.get("temperature_2m", [None])[i],
            "wind_speed_kmh": hourly.get("wind_speed_10m", [None])[i],
            "wind_direction_deg": hourly.get("wind_direction_10m", [None])[i],
            "wind_gusts_kmh": hourly.get("wind_gusts_10m", [None])[i],
            "visibility_m": hourly.get("visibility", [None])[i],
            "cloud_cover_pct": hourly.get("cloud_cover", [None])[i],
            "cloud_cover_low_pct": hourly.get("cloud_cover_low", [None])[i],
            "precipitation_mm": hourly.get("precipitation", [None])[i],
            "pressure_hpa": hourly.get("pressure_msl", [None])[i],
            "weather_code": hourly.get("weather_code", [None])[i],
        })
    return result


def get_current_weather(
    latitude: float = _BLR_LAT,
    longitude: float = _BLR_LON,
) -> Dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,cloud_cover,precipitation,pressure_msl,weather_code",
        "timezone": "Asia/Kolkata",
    }
    resp = requests.get(_OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("current", {})


def get_weather_at_time(
    target_time: datetime,
    latitude: float = _BLR_LAT,
    longitude: float = _BLR_LON,
) -> Optional[Dict[str, Any]]:
    date_str = target_time.strftime("%Y-%m-%d")
    cached = _get_cached_weather(date_str)
    if cached:
        hour = target_time.hour
        for row in cached:
            ts = row["timestamp"]
            if isinstance(ts, str) and f"T{hour:02d}" in ts:
                return dict(row)
    historical = get_historical_weather(date_str, date_str, latitude, longitude)
    if historical:
        _cache_weather(historical)
        hour = target_time.hour
        for row in historical:
            if f"T{hour:02d}" in row.get("timestamp", ""):
                return row
    current = get_current_weather(latitude, longitude)
    return {
        "timestamp": current.get("time", ""),
        "temperature_c": current.get("temperature_2m"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_deg": current.get("wind_direction_10m"),
        "wind_gusts_kmh": current.get("wind_gusts_10m"),
        "visibility_m": None,
        "cloud_cover_pct": current.get("cloud_cover"),
        "cloud_cover_low_pct": None,
        "precipitation_mm": current.get("precipitation"),
        "pressure_hpa": current.get("pressure_msl"),
        "weather_code": current.get("weather_code"),
    }


def _get_cached_weather(date_str: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM weather_cache WHERE timestamp LIKE ?",
            (f"{date_str}%",),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _cache_weather(records: List[Dict[str, Any]], db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    init_weather_table(db_path)
    conn = _connect(path)
    for r in records:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO weather_cache
                (timestamp, temperature_c, wind_speed_kmh, wind_direction_deg,
                 wind_gusts_kmh, visibility_m, cloud_cover_pct, cloud_cover_low_pct,
                 precipitation_mm, pressure_hpa, weather_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    r.get("timestamp"), r.get("temperature_c"), r.get("wind_speed_kmh"),
                    r.get("wind_direction_deg"), r.get("wind_gusts_kmh"),
                    r.get("visibility_m"), r.get("cloud_cover_pct"),
                    r.get("cloud_cover_low_pct"), r.get("precipitation_mm"),
                    r.get("pressure_hpa"), r.get("weather_code"),
                ),
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


def bulk_cache_weather(
    start_date: str,
    end_date: str,
    latitude: float = _BLR_LAT,
    longitude: float = _BLR_LON,
    db_path: Optional[Path] = None,
) -> int:
    records = get_historical_weather(start_date, end_date, latitude, longitude)
    if records:
        _cache_weather(records, db_path)
    return len(records)


def kmh_to_knots(kmh: float) -> float:
    return kmh * 0.539957


def visibility_m_to_sm(m: float) -> float:
    return m / 1609.34
