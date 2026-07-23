from __future__ import annotations

import csv
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data.opensky_client import OpenSkyClient, utc_day_range
from data.weather_client import (
    bulk_cache_weather, get_current_weather, get_weather_at_time,
    init_weather_table, kmh_to_knots,
)


DEFAULT_DB_PATH = Path(__file__).parent / "flights.db"
_CALLSIGN_MAP_PATH = Path(__file__).parent / "callsign_map.csv"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_opensky_tables(db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS opensky_flights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icao24 TEXT NOT NULL,
            callsign TEXT,
            flight_id TEXT,
            origin_airport TEXT,
            destination_airport TEXT,
            first_seen INTEGER,
            last_seen INTEGER,
            duration_min REAL,
            date TEXT,
            aircraft_type TEXT,
            source TEXT DEFAULT 'opensky',
            UNIQUE(icao24, first_seen)
        );

        CREATE TABLE IF NOT EXISTS opensky_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icao24 TEXT NOT NULL,
            callsign TEXT,
            timestamp INTEGER,
            latitude REAL,
            longitude REAL,
            altitude_m REAL,
            velocity_ms REAL,
            heading_deg REAL,
            on_ground INTEGER,
            vertical_rate REAL,
            flight_id TEXT
        );

        CREATE TABLE IF NOT EXISTS rotation_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            icao24 TEXT NOT NULL,
            flight_sequence INTEGER NOT NULL,
            flight_id TEXT,
            callsign TEXT,
            origin TEXT,
            destination TEXT,
            first_seen INTEGER,
            last_seen INTEGER,
            duration_min REAL,
            prev_flight_delay_min REAL,
            date TEXT,
            UNIQUE(icao24, flight_sequence)
        );

        CREATE TABLE IF NOT EXISTS delay_labels (
            flight_id TEXT,
            origin TEXT,
            destination TEXT,
            date TEXT,
            departure_hour INTEGER,
            day_of_week INTEGER,
            actual_duration_min REAL,
            expected_duration_min REAL,
            deviation_min REAL,
            is_delayed INTEGER,
            PRIMARY KEY (flight_id, origin, destination, date)
        );

        CREATE INDEX IF NOT EXISTS idx_os_flights_icao ON opensky_flights(icao24);
        CREATE INDEX IF NOT EXISTS idx_os_flights_date ON opensky_flights(date);
        CREATE INDEX IF NOT EXISTS idx_os_flights_callsign ON opensky_flights(callsign);
        CREATE INDEX IF NOT EXISTS idx_os_states_icao ON opensky_states(icao24);
        CREATE INDEX IF NOT EXISTS idx_rc_icao ON rotation_chains(icao24);
    """)
    conn.commit()
    conn.close()
    init_weather_table(db_path)


def _load_callsign_map() -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not _CALLSIGN_MAP_PATH.exists():
        return mapping
    with open(_CALLSIGN_MAP_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cs = row.get("icao_callsign", "").strip().upper()
            if cs:
                mapping[cs] = {
                    "flight_id": row.get("flight_id", ""),
                    "origin_airport": row.get("origin_airport", ""),
                    "dest_airport": row.get("dest_airport", ""),
                    "aircraft_type": row.get("aircraft_type", ""),
                }
    return mapping


def _resolve_flight(callsign: Optional[str], origin: Optional[str], dest: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cs = (callsign or "").strip().upper()
    mapping = _load_callsign_map()
    if cs in mapping:
        entry = mapping[cs]
        return entry["flight_id"], entry.get("aircraft_type"), cs
    if origin and dest:
        for code, entry in mapping.items():
            if entry.get("origin_airport") == origin and entry.get("dest_airport") == dest:
                return entry["flight_id"], entry.get("aircraft_type"), cs
    return None, None, cs


def store_flights(
    flights: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
    default_origin: Optional[str] = None,
    default_destination: Optional[str] = None,
) -> int:
    path = db_path or DEFAULT_DB_PATH
    init_opensky_tables(db_path)
    conn = _connect(path)
    count = 0
    for f in flights:
        icao24 = f.get("icao24", "")
        first_seen = f.get("firstSeen")
        callsign = (f.get("callsign") or "").strip()
        origin = f.get("estDepartureAirport") or default_origin
        dest = f.get("estArrivalAirport") or default_destination
        flight_id, aircraft_type, _ = _resolve_flight(callsign, origin, dest)
        if not flight_id:
            flight_id = callsign if callsign else f"{icao24}_{first_seen}"
        duration = None
        if first_seen and f.get("lastSeen"):
            duration = round((f["lastSeen"] - first_seen) / 60.0, 1)
        date_str = None
        if first_seen:
            dt = datetime.fromtimestamp(first_seen, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
        try:
            conn.execute(
                """INSERT OR IGNORE INTO opensky_flights
                (icao24, callsign, flight_id, origin_airport, destination_airport,
                 first_seen, last_seen, duration_min, date, aircraft_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (icao24, callsign, flight_id, origin, dest,
                 first_seen, f.get("lastSeen"), duration, date_str, aircraft_type),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return count


def store_states(
    states: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> int:
    path = db_path or DEFAULT_DB_PATH
    init_opensky_tables(db_path)
    conn = _connect(path)
    count = 0
    for s in states:
        icao24 = s.get("icao24", "")
        callsign = (s.get("callsign") or "").strip()
        flight_id = None
        if callsign:
            mapping = _load_callsign_map()
            if callsign.upper() in mapping:
                flight_id = mapping[callsign.upper()]["flight_id"]
        try:
            conn.execute(
                """INSERT INTO opensky_states
                (icao24, callsign, timestamp, latitude, longitude, altitude_m,
                 velocity_ms, heading_deg, on_ground, vertical_rate, flight_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    icao24, callsign,
                    s.get("time_position") or s.get("last_contact"),
                    s.get("latitude"), s.get("longitude"),
                    s.get("baro_altitude"), s.get("velocity"),
                    s.get("true_track"), int(s.get("on_ground", False)),
                    s.get("vertical_rate"), flight_id,
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return count


def compute_rotation_chains(db_path: Optional[Path] = None) -> int:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("DELETE FROM rotation_chains")
    rows = conn.execute(
        """SELECT icao24, callsign, flight_id, origin_airport, destination_airport,
                  first_seen, last_seen, duration_min, date
           FROM opensky_flights
           WHERE first_seen IS NOT NULL
           ORDER BY icao24, first_seen"""
    ).fetchall()
    count = 0
    current_icao = None
    seq = 0
    prev_delay: Optional[float] = None
    prev_last_seen: Optional[int] = None
    for r in rows:
        icao = r["icao24"]
        if icao != current_icao:
            current_icao = icao
            seq = 0
            prev_delay = None
            prev_last_seen = None
        else:
            seq += 1
        expected_gap_min = 45.0
        if prev_last_seen and r["first_seen"]:
            actual_gap_min = (r["first_seen"] - prev_last_seen) / 60.0
            turn_deviation = actual_gap_min - expected_gap_min
            if turn_deviation > 15:
                prev_delay = turn_deviation
            else:
                prev_delay = 0.0
        try:
            conn.execute(
                """INSERT OR IGNORE INTO rotation_chains
                (icao24, flight_sequence, flight_id, callsign, origin, destination,
                 first_seen, last_seen, duration_min, prev_flight_delay_min, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    icao, seq, r["flight_id"], r["callsign"],
                    r["origin_airport"], r["destination_airport"],
                    r["first_seen"], r["last_seen"], r["duration_min"],
                    prev_delay, r["date"],
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
        prev_last_seen = r["last_seen"]
    conn.commit()
    conn.close()
    return count


def compute_delay_labels(
    delay_threshold_min: float = 15.0,
    db_path: Optional[Path] = None,
) -> int:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("DELETE FROM delay_labels")
    rows = conn.execute(
        """SELECT icao24, flight_id, callsign, origin_airport, destination_airport, date,
                  first_seen, duration_min
           FROM opensky_flights
           WHERE duration_min IS NOT NULL AND first_seen IS NOT NULL"""
    ).fetchall()
    route_stats: Dict[str, List[float]] = {}
    for r in rows:
        origin = r["origin_airport"]
        dest = r["destination_airport"]
        if not origin or not dest:
            continue
        key = f"{origin}_{dest}"
        route_stats.setdefault(key, []).append(r["duration_min"])
    route_avgs = {k: sum(v) / len(v) for k, v in route_stats.items() if v}
    count = 0
    for r in rows:
        origin = r["origin_airport"]
        dest = r["destination_airport"]
        if not origin or not dest:
            continue
        key = f"{origin}_{dest}"
        expected = route_avgs.get(key)
        if expected is None:
            expected = r["duration_min"]
        if expected is None:
            continue
        deviation = (r["duration_min"] or 0) - expected
        is_delayed = 1 if deviation > delay_threshold_min else 0
        dt = datetime.fromtimestamp(r["first_seen"], tz=timezone.utc)
        flight_id = r["flight_id"] if r["flight_id"] else f"{r['icao24']}_{r['first_seen']}"
        try:
            conn.execute(
                """INSERT OR IGNORE INTO delay_labels
                (flight_id, origin, destination, date, departure_hour, day_of_week,
                 actual_duration_min, expected_duration_min, deviation_min, is_delayed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    flight_id, origin, dest, r["date"],
                    dt.hour, dt.weekday(),
                    r["duration_min"], round(expected, 1),
                    round(deviation, 1), is_delayed,
                ),
            )
            count += 1
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()
    return count


def seed_historical_data(
    days: int = 7,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    init_opensky_tables(db_path)
    client = OpenSkyClient()
    now = datetime.now(timezone.utc)
    total_flights = 0
    total_weather = 0
    errors = []
    for d in range(days, 0, -1):
        date = now - timedelta(days=d)
        begin, end = utc_day_range(date)
        date_str = date.strftime("%Y-%m-%d")
        try:
            departures = client.get_departures("VOBL", begin, end)
            arrivals = client.get_arrivals("VOBL", begin, end)
            stored_dep = store_flights(departures, default_origin="VOBL", db_path=db_path)
            stored_arr = store_flights(arrivals, default_destination="VOBL", db_path=db_path)
            total_flights += stored_dep + stored_arr
            weather_records = bulk_cache_weather(date_str, date_str, db_path=db_path)
            total_weather += weather_records
        except Exception as e:
            errors.append(f"{date_str}: {e}")
    compute_rotation_chains(db_path)
    compute_delay_labels(db_path=db_path)
    return {
        "status": "success",
        "days_seeded": days,
        "total_flights": total_flights,
        "total_weather_records": total_weather,
        "errors": errors,
        "credits_remaining": client.credits_remaining,
    }


def poll_live_data(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    init_opensky_tables(db_path)
    client = OpenSkyClient()
    try:
        states = client.get_live_states()
        state_count = store_states(states, db_path)
    except Exception as e:
        states = []
        state_count = 0
    deleted = cleanup_old_states(days_to_keep=1, db_path=db_path)
    return {
        "status": "success",
        "live_aircraft": len(states),
        "states_stored": state_count,
        "old_states_cleaned": deleted,
        "credits_remaining": client.credits_remaining,
    }


def get_live_aircraft(db_path: Optional[Path] = None) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()
    conn = _connect(path)
    try:
        rows = conn.execute(
            """SELECT icao24, callsign, flight_id, latitude, longitude,
                      altitude_m, velocity_ms, heading_deg, on_ground, timestamp
               FROM opensky_states
               WHERE timestamp > ?
               ORDER BY timestamp DESC""",
            (int(time.time()) - 3600,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_recent_flights(
    hours: int = 24,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()
    conn = _connect(path)
    cutoff = int(time.time()) - (hours * 3600)
    try:
        rows = conn.execute(
            """SELECT * FROM opensky_flights
               WHERE first_seen > ?
               ORDER BY first_seen DESC""",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def get_rotation_chain(
    icao24: str,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()
    conn = _connect(path)
    try:
        rows = conn.execute(
            """SELECT * FROM rotation_chains
               WHERE icao24 = ?
               ORDER BY flight_sequence""",
            (icao24,),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return pd.DataFrame([dict(r) for r in rows])


def get_feature_table(db_path: Optional[Path] = None) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()
    conn = _connect(path)
    try:
        df = pd.read_sql_query(
            """SELECT
                f.flight_id, f.icao24, f.callsign, f.origin_airport, f.destination_airport,
                f.first_seen, f.last_seen, f.duration_min, f.date, f.aircraft_type,
                rc.prev_flight_delay_min, rc.flight_sequence,
                dl.departure_hour, dl.day_of_week, dl.deviation_min, dl.is_delayed,
                w.temperature_c, w.wind_speed_kmh, w.wind_gusts_kmh,
                w.visibility_m, w.cloud_cover_pct, w.cloud_cover_low_pct,
                w.precipitation_mm, w.pressure_hpa, w.weather_code
               FROM opensky_flights f
               LEFT JOIN rotation_chains rc
                 ON f.icao24 = rc.icao24 AND f.first_seen = rc.first_seen
               LEFT JOIN delay_labels dl
                 ON dl.flight_id = COALESCE(f.flight_id, f.icao24 || '_' || f.first_seen)
                   AND f.origin_airport = dl.origin
                   AND f.destination_airport = dl.destination
               LEFT JOIN weather_cache w
                 ON w.timestamp LIKE f.date || '%' || printf('%02d',
                   CAST((CAST(f.first_seen AS INTEGER) % 86400) / 3600 AS INTEGER))
               WHERE f.duration_min IS NOT NULL""",
            conn,
        )
    except sqlite3.OperationalError:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def cleanup_old_states(days_to_keep: int = 1, db_path: Optional[Path] = None) -> int:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return 0
    conn = _connect(path)
    cutoff = int(time.time()) - (days_to_keep * 86400)
    try:
        cursor = conn.execute("DELETE FROM opensky_states WHERE timestamp < ?", (cutoff,))
        deleted = cursor.rowcount
    except sqlite3.OperationalError:
        deleted = 0
    conn.commit()
    conn.close()
    return deleted


def get_daily_callsigns(
    min_days: int = 3,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            """SELECT callsign,
                      COUNT(DISTINCT date) as days_active,
                      COUNT(*) as total_flights,
                      GROUP_CONCAT(DISTINCT origin_airport) as origins,
                      GROUP_CONCAT(DISTINCT destination_airport) as destinations
               FROM opensky_flights
               WHERE callsign IS NOT NULL AND callsign != ''
                 AND origin_airport IS NOT NULL AND destination_airport IS NOT NULL
               GROUP BY callsign
               HAVING days_active >= ?
               ORDER BY total_flights DESC
               LIMIT ?""",
            (min_days, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_filtered_feature_table(
    callsigns: List[str],
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists() or not callsigns:
        return pd.DataFrame()
    conn = _connect(path)
    placeholders = ",".join("?" for _ in callsigns)
    try:
        df = pd.read_sql_query(
            f"""SELECT
                f.flight_id, f.icao24, f.callsign, f.origin_airport, f.destination_airport,
                f.first_seen, f.last_seen, f.duration_min, f.date, f.aircraft_type,
                rc.prev_flight_delay_min, rc.flight_sequence,
                dl.departure_hour, dl.day_of_week, dl.deviation_min, dl.is_delayed,
                w.temperature_c, w.wind_speed_kmh, w.wind_gusts_kmh,
                w.visibility_m, w.cloud_cover_pct, w.cloud_cover_low_pct,
                w.precipitation_mm, w.pressure_hpa, w.weather_code
               FROM opensky_flights f
               LEFT JOIN rotation_chains rc
                 ON f.icao24 = rc.icao24 AND f.first_seen = rc.first_seen
               LEFT JOIN delay_labels dl
                 ON dl.flight_id = COALESCE(f.flight_id, f.icao24 || '_' || f.first_seen)
                   AND f.origin_airport = dl.origin
                   AND f.destination_airport = dl.destination
               LEFT JOIN weather_cache w
                 ON w.timestamp LIKE f.date || '%' || printf('%02d',
                   CAST((CAST(f.first_seen AS INTEGER) % 86400) / 3600 AS INTEGER))
               WHERE f.duration_min IS NOT NULL
                 AND f.callsign IN ({placeholders})""",
            conn,
            params=callsigns,
        )
    except sqlite3.OperationalError:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def get_flight_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return {"total_flights": 0, "unique_aircraft": 0, "date_range": None}
    conn = _connect(path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM opensky_flights").fetchone()[0]
        aircraft = conn.execute("SELECT COUNT(DISTINCT icao24) FROM opensky_flights").fetchone()[0]
        dates = conn.execute(
            "SELECT MIN(date), MAX(date) FROM opensky_flights WHERE date IS NOT NULL"
        ).fetchone()
        weather_count = conn.execute("SELECT COUNT(*) FROM weather_cache").fetchone()[0]
        states_count = conn.execute("SELECT COUNT(*) FROM opensky_states").fetchone()[0]
    except sqlite3.OperationalError:
        total, aircraft, dates, weather_count, states_count = 0, 0, (None, None), 0, 0
    finally:
        conn.close()
    return {
        "total_flights": total,
        "unique_aircraft": aircraft,
        "date_range": (dates[0], dates[1]) if dates else None,
        "weather_records": weather_count,
        "state_records": states_count,
    }


def get_flight_schedule(
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute("""
            SELECT
                f.callsign,
                f.origin_airport,
                f.destination_airport,
                COUNT(*) as total_flights,
                COUNT(DISTINCT f.date) as days_active,
                ROUND(AVG(
                    CAST((CAST(f.first_seen AS INTEGER) % 86400) / 3600 AS REAL)
                ), 1) as avg_departure_hour,
                ROUND(AVG(f.duration_min), 0) as avg_duration_min,
                SUM(CASE WHEN dl.is_delayed = 1 THEN 1 ELSE 0 END) as delayed_count,
                ROUND(AVG(ABS(COALESCE(dl.deviation_min, 0))), 1) as avg_abs_deviation,
                ROUND(MAX(ABS(COALESCE(dl.deviation_min, 0))), 1) as max_deviation
            FROM opensky_flights f
            LEFT JOIN delay_labels dl
                ON dl.flight_id = COALESCE(f.flight_id, f.callsign, f.icao24 || '_' || f.first_seen)
                AND f.origin_airport = dl.origin
                AND f.destination_airport = dl.destination
            WHERE f.callsign IS NOT NULL AND f.callsign != ''
              AND f.origin_airport IS NOT NULL AND f.destination_airport IS NOT NULL
              AND f.origin_airport != f.destination_airport
              AND f.duration_min IS NOT NULL
            GROUP BY f.callsign, f.origin_airport, f.destination_airport
            HAVING total_flights >= 2
            ORDER BY (CAST(delayed_count AS REAL) / total_flights) * avg_abs_deviation DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    schedule = []
    for r in rows:
        delay_rate = round(r["delayed_count"] / r["total_flights"] * 100, 0) if r["total_flights"] else 0
        hour = r["avg_departure_hour"]
        hour_int = int(hour) if hour else 12
        minute = int((hour - hour_int) * 60) if hour else 0
        schedule.append({
            "callsign": r["callsign"],
            "origin": r["origin_airport"],
            "destination": r["destination_airport"],
            "route": f"{r['origin_airport']}→{r['destination_airport']}",
            "total_flights": r["total_flights"],
            "days_active": r["days_active"],
            "avg_departure_hour": hour_int,
            "avg_departure_minute": minute,
            "avg_duration_min": int(r["avg_duration_min"] or 0),
            "delayed_count": r["delayed_count"],
            "delay_rate_pct": int(delay_rate),
            "avg_deviation_min": r["avg_abs_deviation"],
            "max_deviation_min": r["max_deviation"],
            "anomaly_score": round(delay_rate * r["avg_abs_deviation"] / 100, 1) if r["avg_abs_deviation"] else 0,
        })
    return schedule


def get_today_schedule(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    from ml_engine.delay_predictor import predict_delay

    schedule = get_flight_schedule(limit=20, db_path=db_path)
    today = datetime.now(timezone.utc)

    for flight in schedule:
        dep_hour = flight["avg_departure_hour"]
        dep_minute = flight["avg_departure_minute"]
        duration = flight["avg_duration_min"]

        scheduled_dep = today.replace(hour=dep_hour, minute=dep_minute, second=0, microsecond=0)
        scheduled_arr = scheduled_dep + timedelta(minutes=duration)

        weather = {}
        try:
            weather = get_weather_at_time(scheduled_dep)
        except Exception:
            pass

        pred = predict_delay(
            origin=flight["origin"],
            destination=flight["destination"],
            departure_hour=dep_hour,
            wind_speed_kmh=weather.get("wind_speed_kmh") or 0,
            wind_gusts_kmh=weather.get("wind_gusts_kmh") or 0,
            visibility_m=weather.get("visibility_m") or 10000,
            cloud_cover_pct=weather.get("cloud_cover_pct") or 0,
            precipitation_mm=weather.get("precipitation_mm") or 0,
            temperature_c=weather.get("temperature_c") or 25,
            pressure_hpa=weather.get("pressure_hpa") or 1013,
        )

        flight["scheduled_departure"] = scheduled_dep.strftime("%H:%M")
        flight["scheduled_arrival"] = scheduled_arr.strftime("%H:%M")
        flight["weather"] = {
            "temp_c": weather.get("temperature_c", "N/A"),
            "wind_kmh": weather.get("wind_speed_kmh", "N/A"),
            "visibility_m": weather.get("visibility_m", "N/A"),
            "precipitation_mm": weather.get("precipitation_mm", 0),
        }
        flight["prediction"] = {
            "delay_probability": pred["delay_probability"],
            "expected_delay_min": pred["expected_delay_min"],
            "risk_level": pred["risk_level"],
            "factors": pred.get("factors", []),
            "model_used": pred["model_used"],
        }

    return schedule
