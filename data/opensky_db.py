from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data.opensky_client import OpenSkyClient, utc_day_range
from data.weather_client import (
    bulk_cache_weather, get_current_weather, get_weather_at_time,
    get_historical_weather, get_forecast_weather, _cache_weather,
    _get_cached_weather, init_weather_table, kmh_to_knots,
)


DEFAULT_DB_PATH = Path(__file__).parent / "flights.db"
_CALLSIGN_MAP_PATH = Path(__file__).parent / "callsign_map.csv"

SCHEDULE_LIMIT = 20
DELAY_RESERVED_SLOTS = 10
MIN_DELAY_SAMPLES = 5


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return conn


def _connect_write(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return conn


def _retry_on_lock(fn, max_retries=10, delay=0.5):
    for attempt in range(max_retries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "database is locked" not in str(e):
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(min(delay * (attempt + 1), 3))


def init_opensky_tables(db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect_write(path)
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

        CREATE TABLE IF NOT EXISTS realtime_delays (
            flight_id TEXT,
            callsign TEXT PRIMARY KEY,
            status TEXT,
            scheduled_departure TEXT,
            actual_departure INTEGER,
            delay_minutes INTEGER,
            last_seen INTEGER,
            detected_at INTEGER,
            origin TEXT,
            destination TEXT,
            at_risk INTEGER DEFAULT 0,
            risk_reason TEXT
        );

        CREATE TABLE IF NOT EXISTS prediction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            callsign TEXT NOT NULL,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            date TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            delay_probability REAL,
            expected_delay_min REAL,
            risk_level TEXT,
            actual_delay_min REAL,
            actual_is_delayed INTEGER,
            actual_recorded_at TEXT,
            UNIQUE(callsign, origin, destination, date)
        );

        CREATE TABLE IF NOT EXISTS prediction_cache (
            callsign TEXT NOT NULL,
            date TEXT NOT NULL,
            origin TEXT,
            destination TEXT,
            departure_hour INTEGER,
            weekday_weight REAL,
            delay_probability REAL,
            expected_delay_min REAL,
            risk_level TEXT,
            factors TEXT,
            model_used TEXT,
            created_at TEXT,
            PRIMARY KEY (callsign, date)
        );

        CREATE INDEX IF NOT EXISTS idx_pl_date ON prediction_log(date);
        CREATE INDEX IF NOT EXISTS idx_pl_callsign ON prediction_log(callsign);
    """)
    try:
        conn.execute("ALTER TABLE prediction_cache ADD COLUMN weekday_weight REAL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
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
    delay_threshold_pct: float = 0.20,
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

    route_hours: Dict[str, Dict[int, List[float]]] = {}
    for r in rows:
        origin = r["origin_airport"]
        dest = r["destination_airport"]
        if not origin or not dest:
            continue
        key = f"{origin}_{dest}"
        dt = datetime.fromtimestamp(r["first_seen"], tz=timezone.utc)
        hour_bucket = dt.hour // 3 * 3
        route_hours.setdefault(key, {}).setdefault(hour_bucket, []).append(r["duration_min"])

    route_expected: Dict[str, Dict[int, float]] = {}
    for key, hours in route_hours.items():
        route_expected[key] = {}
        for hour_bucket, durations in hours.items():
            route_expected[key][hour_bucket] = sum(durations) / len(durations)

    route_avgs: Dict[str, float] = {}
    for key, hours in route_hours.items():
        all_durations = [d for bucket_durs in hours.values() for d in bucket_durs]
        if all_durations:
            route_avgs[key] = sum(all_durations) / len(all_durations)

    count = 0
    for r in rows:
        origin = r["origin_airport"]
        dest = r["destination_airport"]
        if not origin or not dest:
            continue
        key = f"{origin}_{dest}"
        dt = datetime.fromtimestamp(r["first_seen"], tz=timezone.utc)
        hour_bucket = dt.hour // 3 * 3

        expected = None
        if key in route_expected and hour_bucket in route_expected[key]:
            expected = route_expected[key][hour_bucket]
        if expected is None:
            expected = route_avgs.get(key)
        if expected is None:
            expected = r["duration_min"]
        if expected is None:
            continue

        deviation = (r["duration_min"] or 0) - expected
        threshold = max(expected * delay_threshold_pct, 10.0)
        is_delayed = 1 if deviation > threshold else 0

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

    conn = _connect(path)
    try:
        existing_dates = set(
            row[0] for row in conn.execute(
                "SELECT DISTINCT date FROM opensky_flights WHERE date IS NOT NULL"
            ).fetchall()
        )
    except Exception:
        existing_dates = set()
    finally:
        conn.close()

    days_skipped = 0
    for d in range(days, 0, -1):
        date = now - timedelta(days=d)
        date_str = date.strftime("%Y-%m-%d")

        if date_str in existing_dates:
            days_skipped += 1
            continue

        begin, end = utc_day_range(date)
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

    if total_flights > 0:
        compute_rotation_chains(db_path)
        compute_delay_labels(db_path=db_path)

    return {
        "status": "success",
        "days_seeded": days,
        "days_skipped": days_skipped,
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

    delay_events = 0
    try:
        from data.live_delay_detector import detect_round_trip_delays, store_realtime_delays
        delays = detect_round_trip_delays(db_path=path)
        store_realtime_delays(delays, db_path=path)
        delay_events = len([d for d in delays if d.get("delay_minutes") is not None])
    except Exception:
        pass

    return {
        "status": "success",
        "live_aircraft": len(states),
        "states_stored": state_count,
        "old_states_cleaned": deleted,
        "delay_events": delay_events,
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


def get_states_for_callsigns(
    callsigns: List[str],
    hours: int = 2,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists() or not callsigns:
        return pd.DataFrame()
    conn = _connect(path)
    cutoff = int(time.time()) - (hours * 3600)
    placeholders = ",".join("?" for _ in callsigns)
    try:
        df = pd.read_sql_query(
            f"""SELECT icao24, callsign, timestamp, latitude, longitude,
                       altitude_m, velocity_ms, heading_deg, on_ground,
                       vertical_rate, flight_id
                FROM opensky_states
                WHERE UPPER(callsign) IN ({placeholders})
                  AND timestamp > ?
                ORDER BY callsign, timestamp""",
            conn,
            params=[c.upper() for c in callsigns] + [cutoff],
        )
    except sqlite3.OperationalError:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


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
                   AND f.date = dl.date
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


def get_route_dayofweek_stats(
    db_path: Optional[Path] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return {}
    conn = _connect(path)
    try:
        rows = conn.execute("""
            SELECT origin, destination, day_of_week,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed,
                   ROUND(AVG(ABS(COALESCE(deviation_min, 0))), 1) as avg_deviation
            FROM delay_labels
            WHERE origin IS NOT NULL AND destination IS NOT NULL
            GROUP BY origin, destination, day_of_week
            HAVING total >= 2
        """).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    stats: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        route = f"{r['origin']}_{r['destination']}"
        dow = r["day_of_week"]
        stats.setdefault(route, {})[dow] = {
            "delay_rate": round(r["delayed"] / r["total"], 4) if r["total"] > 0 else 0,
            "avg_deviation": r["avg_deviation"] or 0,
            "count": r["total"],
        }
    return stats


def get_flight_schedule(
    limit: int = 20,
    db_path: Optional[Path] = None,
    day_of_week: Optional[int] = None,
    reference_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        if day_of_week is not None:
            ref = reference_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rows = conn.execute("""
                SELECT callsign, origin_airport, destination_airport,
                       total_flights, days_active,
                       avg_departure_hour, avg_duration_min,
                       delayed_w, weighted_count, dow_sample_count,
                       delayed_count, wdev_sum, max_deviation
                FROM (
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
                        SUM(CASE WHEN dl.is_delayed = 1 THEN w ELSE 0 END) as delayed_w,
                        SUM(w) as weighted_count,
                        COUNT(dl.flight_id) as dow_sample_count,
                        SUM(CASE WHEN dl.is_delayed = 1 THEN 1 ELSE 0 END) as delayed_count,
                        SUM(ABS(COALESCE(dl.deviation_min, 0)) * w) as wdev_sum,
                        ROUND(MAX(ABS(COALESCE(dl.deviation_min, 0))), 1) as max_deviation
                    FROM opensky_flights f
                       LEFT JOIN delay_labels dl
                         ON dl.flight_id = COALESCE(f.flight_id, f.icao24 || '_' || f.first_seen)
                           AND f.origin_airport = dl.origin
                           AND f.destination_airport = dl.destination
                           AND f.date = dl.date
                           AND dl.day_of_week = ?
                       LEFT JOIN (
                           SELECT DISTINCT date,
                                  1.0 / (1.0 + 0.5 * ((julianday(?) - julianday(date)) / 7.0)) AS w
                           FROM delay_labels
                           WHERE date IS NOT NULL
                       ) w
                         ON w.date = dl.date AND dl.flight_id IS NOT NULL
                    WHERE f.callsign IS NOT NULL AND f.callsign != ''
                      AND f.origin_airport IS NOT NULL AND f.destination_airport IS NOT NULL
                      AND f.origin_airport != f.destination_airport
                      AND f.duration_min IS NOT NULL
                    GROUP BY f.callsign, f.origin_airport, f.destination_airport
                    HAVING total_flights >= 2
                )
                ORDER BY total_flights DESC, days_active DESC
            """, (day_of_week, ref)).fetchall()
        else:
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
                     ON dl.flight_id = COALESCE(f.flight_id, f.icao24 || '_' || f.first_seen)
                       AND f.origin_airport = dl.origin
                       AND f.destination_airport = dl.destination
                       AND f.date = dl.date
                WHERE f.callsign IS NOT NULL AND f.callsign != ''
                  AND f.origin_airport IS NOT NULL AND f.destination_airport IS NOT NULL
                  AND f.origin_airport != f.destination_airport
                  AND f.duration_min IS NOT NULL
                GROUP BY f.callsign, f.origin_airport, f.destination_airport
                HAVING total_flights >= 2
                ORDER BY total_flights DESC, days_active DESC
            """).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    delay_slots = min(DELAY_RESERVED_SLOTS, limit)
    if rows and delay_slots > 0:
        def _rank_key(r, by_delay):
            if by_delay:
                return (r["delayed_count"] or 0, r["total_flights"] or 0, r["days_active"] or 0, r["callsign"])
            return (r["total_flights"] or 0, r["days_active"] or 0, r["callsign"])

        delayed_sorted = sorted(rows, key=lambda r: _rank_key(r, True), reverse=True)
        volume_sorted = sorted(rows, key=lambda r: _rank_key(r, False), reverse=True)

        chosen = []
        seen = set()
        picked = 0
        for r in delayed_sorted:
            if picked >= delay_slots:
                break
            key = (r["callsign"], r["origin_airport"], r["destination_airport"])
            if (r["delayed_count"] or 0) > 0 and (r["total_flights"] or 0) >= MIN_DELAY_SAMPLES and key not in seen:
                seen.add(key)
                chosen.append(r)
                picked += 1
        for r in volume_sorted:
            if len(chosen) >= limit:
                break
            key = (r["callsign"], r["origin_airport"], r["destination_airport"])
            if key not in seen:
                seen.add(key)
                chosen.append(r)
        rows = chosen

    overall_stats: Dict[str, Dict[str, Any]] = {}
    if day_of_week is not None:
        try:
            conn2 = _connect(path)
            overall_rows = conn2.execute("""
                SELECT origin, destination,
                       COUNT(*) as total,
                       SUM(CASE WHEN is_delayed = 1 THEN 1 ELSE 0 END) as delayed
                FROM delay_labels
                WHERE origin IS NOT NULL AND destination IS NOT NULL
                GROUP BY origin, destination
                HAVING total >= 2
            """).fetchall()
            conn2.close()
            for r in overall_rows:
                key = f"{r['origin']}_{r['destination']}"
                overall_stats[key] = {
                    "delay_rate": round(r["delayed"] / r["total"], 4) if r["total"] > 0 else 0,
                    "count": r["total"],
                }
        except sqlite3.OperationalError:
            pass

    schedule = []
    for r in rows:
        if day_of_week is not None:
            weighted_count = r["weighted_count"] or 0
            delayed_w = r["delayed_w"] or 0
            delay_rate = round(delayed_w / weighted_count * 100, 0) if weighted_count else 0
            avg_deviation = round(r["wdev_sum"] / weighted_count, 1) if weighted_count else 0
            dow_sample_count = r["dow_sample_count"] or 0
            dow_weighted_count = round(weighted_count, 1)
            delayed_count = round(delayed_w, 2)
        else:
            delay_rate = round(r["delayed_count"] / r["total_flights"] * 100, 0) if r["total_flights"] else 0
            avg_deviation = r["avg_abs_deviation"]
            dow_sample_count = None
            dow_weighted_count = None
            delayed_count = r["delayed_count"]
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
            "delayed_count": delayed_count,
            "delay_rate_pct": int(delay_rate),
            "avg_deviation_min": avg_deviation,
            "max_deviation_min": r["max_deviation"],
            "anomaly_score": round(delay_rate * avg_deviation / 100, 1) if avg_deviation else 0,
            "dow_delay_rate": round(delay_rate / 100, 4),
            "overall_delay_rate": overall_stats.get(f"{r['origin_airport']}_{r['destination_airport']}", {}).get("delay_rate", round(delay_rate / 100, 4)),
            "dow_sample_count": dow_sample_count,
            "dow_weighted_count": dow_weighted_count,
            "flight_sequence": 0,
            "prev_flight_delay_min": None,
            "is_return_leg": False,
        })

    if schedule:
        callsigns = [f["callsign"] for f in schedule]
        placeholders = ",".join("?" for _ in callsigns)
        try:
            conn2 = _connect(path)
            rc_rows = conn2.execute(
                f"""SELECT rc.callsign, rc.flight_sequence, rc.prev_flight_delay_min,
                           rc.origin, rc.destination, rc.date,
                           f.icao24
                    FROM rotation_chains rc
                    JOIN opensky_flights f
                      ON rc.icao24 = f.icao24 AND rc.first_seen = f.first_seen
                    WHERE rc.callsign IN ({placeholders})
                    ORDER BY rc.date DESC, rc.callsign, rc.flight_sequence""",
                callsigns,
            ).fetchall()
            conn2.close()

            seen_callsigns: set = set()
            for rc_row in rc_rows:
                cs = rc_row["callsign"]
                if cs in seen_callsigns:
                    continue
                seen_callsigns.add(cs)
                for flight in schedule:
                    if flight["callsign"] == cs:
                        flight["flight_sequence"] = rc_row["flight_sequence"]
                        flight["prev_flight_delay_min"] = rc_row["prev_flight_delay_min"]
                        flight["is_return_leg"] = rc_row["flight_sequence"] > 0
                        break
        except sqlite3.OperationalError:
            pass

    return schedule


def get_model_flights_with_status(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    schedule = get_flight_schedule(limit=SCHEDULE_LIMIT, db_path=db_path)
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return schedule

    icao_map: Dict[str, Dict[str, Any]] = {}
    conn = _connect(path)
    try:
        callsigns = [f["callsign"] for f in schedule]
        if not callsigns:
            conn.close()
            return schedule
        placeholders = ",".join("?" for _ in callsigns)
        rows = conn.execute(
            f"""SELECT callsign, icao24,
                       MAX(first_seen) as latest_first_seen
                FROM opensky_flights
                WHERE callsign IN ({placeholders})
                GROUP BY callsign""",
            callsigns,
        ).fetchall()
        for r in rows:
            icao_map[r["callsign"]] = {
                "icao24": r["icao24"],
                "latest_first_seen": r["latest_first_seen"],
            }
    except sqlite3.OperationalError:
        pass

    icao24s = list({v["icao24"] for v in icao_map.values() if v.get("icao24")})
    state_map: Dict[str, List[Dict]] = {}
    if icao24s:
        placeholders = ",".join("?" for _ in icao24s)
        try:
            rows = conn.execute(
                f"""SELECT icao24, callsign, timestamp, altitude_m, velocity_ms,
                           on_ground, vertical_rate
                    FROM opensky_states
                    WHERE icao24 IN ({placeholders})
                    ORDER BY timestamp ASC""",
                icao24s,
            ).fetchall()
            for r in rows:
                state_map.setdefault(r["icao24"], []).append(dict(r))
        except sqlite3.OperationalError:
            pass

    all_rotation: List[Dict] = []
    try:
        all_rotation = [dict(r) for r in conn.execute(
            """SELECT icao24, callsign, origin, destination, date, flight_sequence
               FROM rotation_chains
               WHERE origin IS NOT NULL AND destination IS NOT NULL
               ORDER BY date, icao24, flight_sequence"""
        ).fetchall()]
    except sqlite3.OperationalError:
        pass
    conn.close()

    flights_by_date: Dict[str, List[Dict]] = {}
    for cr in all_rotation:
        flights_by_date.setdefault(cr["date"], []).append(cr)

    now = int(time.time())
    today = datetime.now(timezone.utc)

    for flight in schedule:
        callsign = flight["callsign"]
        info = icao_map.get(callsign, {})
        icao24 = info.get("icao24", "")
        states = state_map.get(icao24, [])
        orig = flight["origin"]
        dest = flight["destination"]

        has_return = False
        return_callsign = ""
        for _d, day_flights in flights_by_date.items():
            for cr in day_flights:
                if cr["origin"] == orig and cr["destination"] == dest:
                    for later in day_flights:
                        if (later["flight_sequence"] > cr["flight_sequence"]
                                and later["origin"] == dest
                                and later["destination"] == "VOBL"
                                and later["icao24"] == cr["icao24"]):
                            has_return = True
                            return_callsign = later["callsign"] or ""
                            break
                if has_return:
                    break
            if has_return:
                break

        if not has_return:
            for _d, day_flights in flights_by_date.items():
                for cr in day_flights:
                    if cr["origin"] == dest and cr["destination"] == orig:
                        has_return = True
                        return_callsign = cr["callsign"] or ""
                        break
                if has_return:
                    break

        flight["flight_type"] = "Round-trip" if has_return else "One-way"
        flight["return_callsign"] = return_callsign

        dep_hour = flight["avg_departure_hour"]
        dep_minute = flight["avg_departure_minute"]
        duration = flight["avg_duration_min"]

        scheduled_dep = today.replace(
            hour=dep_hour, minute=dep_minute, second=0, microsecond=0
        )
        scheduled_dep_ts = int(scheduled_dep.timestamp())

        flight["scheduled_departure"] = scheduled_dep.strftime("%H:%M")
        flight["scheduled_arrival"] = (
            scheduled_dep + timedelta(minutes=duration)
        ).strftime("%H:%M")

        if not states:
            if now < scheduled_dep_ts:
                flight["status"] = "Scheduled"
            else:
                flight["status"] = "Unknown"
            flight["delay_minutes"] = None
            flight["notes"] = ""
            if has_return:
                ret_tag = f" ({return_callsign})" if return_callsign else ""
                flight["notes"] = f"Return leg{ret_tag} likely affected if delayed"
            continue

        latest = states[-1]
        on_ground = bool(latest["on_ground"])
        vel = latest.get("velocity_ms") or 0

        departed_ts = None
        landed_ts = None
        for i in range(1, len(states)):
            prev_st = states[i - 1]
            curr_st = states[i]
            if prev_st["on_ground"] and not curr_st["on_ground"] and (curr_st.get("altitude_m") or 0) > 30:
                departed_ts = curr_st["timestamp"]
            if not prev_st["on_ground"] and curr_st["on_ground"]:
                landed_ts = curr_st["timestamp"]

        if landed_ts:
            flight["status"] = "Landed"
            landed_dt = datetime.fromtimestamp(landed_ts, tz=timezone.utc)
            flight["notes"] = f"Landed {landed_dt.strftime('%H:%M')} UTC"
        elif departed_ts:
            if on_ground:
                flight["status"] = "Landed"
                flight["notes"] = "On ground after flight"
            else:
                flight["status"] = "In Air"
                eta_ts = departed_ts + (duration * 60)
                eta_dt = datetime.fromtimestamp(eta_ts, tz=timezone.utc)
                remaining_min = max(0, round((eta_ts - now) / 60))
                if remaining_min > 0:
                    flight["notes"] = f"ETA {eta_dt.strftime('%H:%M')} UTC ({remaining_min}min left)"
                else:
                    flight["notes"] = f"ETA was {eta_dt.strftime('%H:%M')} UTC"
        elif on_ground:
            if vel > 2:
                flight["status"] = "Taxiing"
                flight["notes"] = "Departing"
            else:
                flight["status"] = "At Gate"
                if now < scheduled_dep_ts:
                    mins_to_dep = round((scheduled_dep_ts - now) / 60)
                    flight["notes"] = f"Departs in {mins_to_dep}min"
                else:
                    flight["notes"] = "Awaiting departure"
        else:
            flight["status"] = "In Air"
            flight["notes"] = ""

        if departed_ts:
            delay = round((departed_ts - scheduled_dep_ts) / 60)
            flight["delay_minutes"] = delay
        else:
            flight["delay_minutes"] = None

        if has_return and flight.get("delay_minutes") and flight["delay_minutes"] > 0:
            ret_tag = f" ({return_callsign})" if return_callsign else ""
            flight["notes"] += f" | Return leg{ret_tag} likely delayed too"

    return schedule


def _get_cached_prediction(
    callsign: str,
    date: str,
    weekday_weight: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return None
    try:
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT delay_probability, expected_delay_min, risk_level, factors, model_used, weekday_weight "
                "FROM prediction_cache WHERE callsign = ? AND date = ?",
                (callsign, date),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    if weekday_weight is not None:
        if row["weekday_weight"] is None or abs(row["weekday_weight"] - weekday_weight) > 0.001:
            return None
    try:
        factors = json.loads(row["factors"]) if row["factors"] else []
    except Exception:
        factors = []
    return {
        "delay_probability": row["delay_probability"],
        "expected_delay_min": row["expected_delay_min"],
        "risk_level": row["risk_level"],
        "factors": factors,
        "model_used": row["model_used"],
    }


def _save_cached_prediction(
    callsign: str,
    date: str,
    origin: str,
    destination: str,
    departure_hour: int,
    pred: Dict[str, Any],
    weekday_weight: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> None:
    path = db_path or DEFAULT_DB_PATH
    try:
        conn = _connect_write(path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO prediction_cache
                (callsign, date, origin, destination, departure_hour,
                 weekday_weight,
                 delay_probability, expected_delay_min, risk_level, factors, model_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    callsign, date, origin, destination, departure_hour,
                    weekday_weight,
                    pred.get("delay_probability"), pred.get("expected_delay_min"),
                    pred.get("risk_level"),
                    json.dumps(pred.get("factors", [])),
                    pred.get("model_used"),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        pass


def get_today_schedule(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    from ml_engine.delay_predictor import predict_delay, _load_models, WEEKDAY_WEIGHT_DEFAULT

    path = db_path or DEFAULT_DB_PATH
    today = datetime.now(timezone.utc)
    schedule = get_flight_schedule(limit=SCHEDULE_LIMIT, db_path=path, day_of_week=today.weekday())

    _load_models()

    cached_weather: Dict[str, Dict] = {}
    today_str = today.strftime("%Y-%m-%d")

    def _index_weather(records) -> None:
        for r in records:
            ts = r.get("timestamp", "")
            if isinstance(ts, str):
                for h in range(24):
                    if f"T{h:02d}" in ts:
                        cached_weather[f"{today_str}-{h:02d}"] = r
                        break

    try:
        existing = _get_cached_weather(today_str, path)
        if existing:
            _index_weather([dict(r) for r in existing])
        else:
            records = get_forecast_weather(today_str, today_str)
            if records:
                _cache_weather(records, path)
                _index_weather(records)
    except Exception:
        pass

    for flight in schedule:
        dep_hour = flight["avg_departure_hour"]
        dep_minute = flight["avg_departure_minute"]
        duration = flight["avg_duration_min"]
        is_return = flight.get("is_return_leg", False)
        fid = flight["callsign"]

        scheduled_dep = today.replace(hour=dep_hour, minute=dep_minute, second=0, microsecond=0)
        scheduled_arr = scheduled_dep + timedelta(minutes=duration)

        weather_key = scheduled_dep.strftime("%Y-%m-%d-%H")
        weather = cached_weather.get(weather_key, {})
        if not weather:
            try:
                weather = get_weather_at_time(scheduled_dep)
            except Exception:
                pass

        cached_pred = _get_cached_prediction(fid, today_str, WEEKDAY_WEIGHT_DEFAULT, path)
        if cached_pred:
            pred = cached_pred
        else:
            pred = predict_delay(
                origin=flight["origin"],
                destination=flight["destination"],
                departure_hour=dep_hour,
                departure_time=scheduled_dep,
                wind_speed_kmh=weather.get("wind_speed_kmh") or 0,
                wind_gusts_kmh=weather.get("wind_gusts_kmh") or 0,
                visibility_m=weather.get("visibility_m") or 10000,
                cloud_cover_pct=weather.get("cloud_cover_pct") or 0,
                precipitation_mm=weather.get("precipitation_mm") or 0,
                temperature_c=weather.get("temperature_c") or 25,
                pressure_hpa=weather.get("pressure_hpa") or 1013,
                delay_rate_pct=flight.get("delay_rate_pct"),
                dow_delay_rate=flight.get("dow_delay_rate"),
                overall_delay_rate=flight.get("overall_delay_rate"),
                dow_sample_count=flight.get("dow_sample_count"),
            )
            _save_cached_prediction(
                fid, today_str,
                origin=flight["origin"], destination=flight["destination"],
                departure_hour=dep_hour, pred=pred,
                weekday_weight=WEEKDAY_WEIGHT_DEFAULT, db_path=path,
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

        if is_return:
            flight["leg_type"] = "Return Leg"
            prev_delay = flight.get("prev_flight_delay_min") or 0
            turnaround_min = 45 + prev_delay
            flight["layover_turnaround_min"] = int(turnaround_min)
            flight["crew_action"] = (
                f"Layover crew change needed after arrival at {scheduled_arr.strftime('%H:%M')} UTC "
                f"({int(turnaround_min)}min turnaround) — fresh crew must board before next departure"
            )
        else:
            flight["leg_type"] = "First Leg"
            flight["layover_turnaround_min"] = None
            flight["crew_action"] = (
                f"Pre-departure crew assignment required before {scheduled_dep.strftime('%H:%M')} UTC"
            )

    return schedule


def get_schedule_for_date(
    target_date,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    from ml_engine.delay_predictor import predict_delay, _load_models

    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    schedule = get_flight_schedule(
        limit=SCHEDULE_LIMIT, db_path=db_path, day_of_week=dt.weekday(),
        reference_date=dt.strftime("%Y-%m-%d"),
    )

    _load_models()

    cached_weather: Dict[str, Dict] = {}
    dt_str = dt.strftime("%Y-%m-%d")
    try:
        records = get_historical_weather(dt_str, dt_str)
        if not records:
            records = get_forecast_weather(dt_str, dt_str)
        if records:
            _cache_weather(records)
            for r in records:
                ts = r.get("timestamp", "")
                if isinstance(ts, str):
                    for h in range(24):
                        if f"T{h:02d}" in ts:
                            cached_weather[f"{dt_str}-{h:02d}"] = r
                            break
    except Exception:
        pass

    for flight in schedule:
        dep_hour = flight["avg_departure_hour"]
        dep_minute = flight["avg_departure_minute"]
        duration = flight["avg_duration_min"]
        is_return = flight.get("is_return_leg", False)

        scheduled_dep = dt.replace(hour=dep_hour, minute=dep_minute, second=0, microsecond=0)
        scheduled_arr = scheduled_dep + timedelta(minutes=duration)

        weather_key = scheduled_dep.strftime("%Y-%m-%d-%H")
        weather = cached_weather.get(weather_key, {})
        if not weather:
            try:
                weather = get_weather_at_time(scheduled_dep)
            except Exception:
                pass

        pred = predict_delay(
            origin=flight["origin"],
            destination=flight["destination"],
            departure_hour=dep_hour,
            departure_time=scheduled_dep,
            wind_speed_kmh=weather.get("wind_speed_kmh") or 0,
            wind_gusts_kmh=weather.get("wind_gusts_kmh") or 0,
            visibility_m=weather.get("visibility_m") or 10000,
            cloud_cover_pct=weather.get("cloud_cover_pct") or 0,
            precipitation_mm=weather.get("precipitation_mm") or 0,
            temperature_c=weather.get("temperature_c") or 25,
            pressure_hpa=weather.get("pressure_hpa") or 1013,
            delay_rate_pct=flight.get("delay_rate_pct"),
            dow_delay_rate=flight.get("dow_delay_rate"),
            overall_delay_rate=flight.get("overall_delay_rate"),
            dow_sample_count=flight.get("dow_sample_count"),
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

        if is_return:
            flight["leg_type"] = "Return Leg"
            prev_delay = flight.get("prev_flight_delay_min") or 0
            turnaround_min = 45 + prev_delay
            flight["layover_turnaround_min"] = int(turnaround_min)
            flight["crew_action"] = (
                f"Layover crew change needed after arrival at {scheduled_arr.strftime('%H:%M')} UTC "
                f"({int(turnaround_min)}min turnaround) — fresh crew must board before next departure"
            )
        else:
            flight["leg_type"] = "First Leg"
            flight["layover_turnaround_min"] = None
            flight["crew_action"] = (
                f"Pre-departure crew assignment required before {scheduled_dep.strftime('%H:%M')} UTC"
            )

    return schedule


def get_data_date_range(
    db_path: Optional[Path] = None,
) -> Tuple[str, str]:
    """Return (min_date, max_date) for which flight data exists, as YYYY-MM-DD strings."""
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return ("", "")
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT MIN(date) as lo, MAX(date) as hi FROM opensky_flights WHERE date IS NOT NULL"
        ).fetchone()
        return (row["lo"] or "", row["hi"] or "")
    except sqlite3.OperationalError:
        return ("", "")
    finally:
        conn.close()


def get_flight_actuals_for_date(
    target_date,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Expected vs actual times for the scheduled flights on a given date.

    Frame is the recurring schedule (flights table). Scheduled/expected departure
    uses the flight's typical departure time (same avg used across the app).
    Actual clock times come from opensky_flights (ADS-B first/last seen),
    expected/actual duration and delay status from delay_labels, and
    prediction-vs-outcome where the model logged a prediction for that date.
    """
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
    date_str = target_date.strftime("%Y-%m-%d")
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            """
            SELECT
                fl.flight_id AS callsign,
                fl.origin,
                fl.destination,
                fl.aircraft_type,
                fl.flight_duration_min,
                ROUND(AVG(
                    CAST((CAST(fh.first_seen AS INTEGER) % 86400) / 3600.0 AS REAL)
                ), 1) AS avg_departure_hour,
                osf.first_seen,
                osf.last_seen,
                dl.expected_duration_min,
                dl.actual_duration_min,
                dl.deviation_min,
                dl.is_delayed,
                pl.delay_probability,
                pl.expected_delay_min AS predicted_delay_min,
                pl.risk_level AS predicted_risk,
                pl.actual_delay_min,
                pl.actual_is_delayed
            FROM flights fl
            LEFT JOIN opensky_flights fh
              ON fh.callsign = fl.flight_id
             AND fh.origin_airport = fl.origin
             AND fh.destination_airport = fl.destination
            LEFT JOIN (
                SELECT callsign, origin_airport, destination_airport, date,
                       MIN(first_seen) AS first_seen, MIN(last_seen) AS last_seen
                FROM opensky_flights
                WHERE date = ?
                GROUP BY callsign, origin_airport, destination_airport
            ) osf
              ON osf.callsign = fl.flight_id
             AND osf.origin_airport = fl.origin
             AND osf.destination_airport = fl.destination
            LEFT JOIN delay_labels dl
              ON dl.flight_id = fl.flight_id
             AND dl.origin = fl.origin
             AND dl.destination = fl.destination
             AND dl.date = ?
            LEFT JOIN prediction_log pl
              ON pl.callsign = fl.flight_id
             AND pl.origin = fl.origin
             AND pl.destination = fl.destination
             AND pl.date = ?
             AND pl.id = (
                 SELECT MAX(id) FROM prediction_log pl2
                 WHERE pl2.callsign = fl.flight_id
                   AND pl2.origin = fl.origin
                   AND pl2.destination = fl.destination
                   AND pl2.date = ?
             )
            GROUP BY fl.flight_id, fl.origin, fl.destination, fl.aircraft_type,
                     fl.flight_duration_min, osf.first_seen, osf.last_seen,
                     dl.expected_duration_min, dl.actual_duration_min,
                     dl.deviation_min, dl.is_delayed,
                     pl.delay_probability, pl.expected_delay_min, pl.risk_level,
                     pl.actual_delay_min, pl.actual_is_delayed
            ORDER BY fl.std
            """,
            (date_str, date_str, date_str, date_str),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    actuals = []
    for r in rows:
        scheduled_dep = None
        scheduled_arr = None
        hour = r["avg_departure_hour"]
        if hour is not None:
            hour_int = int(hour)
            minute = int((hour - hour_int) * 60)
            scheduled_dep = f"{hour_int:02d}:{minute:02d}"
            expected_min = r["expected_duration_min"]
            if expected_min is None:
                expected_min = r["flight_duration_min"] or 0
            dep_dt = datetime.now(timezone.utc).replace(hour=hour_int, minute=minute, second=0, microsecond=0)
            scheduled_arr = (dep_dt + timedelta(minutes=expected_min)).strftime("%H:%M")
        actual_dep = None
        actual_arr = None
        if r["first_seen"] is not None:
            actual_dep = datetime.fromtimestamp(r["first_seen"], timezone.utc).strftime("%H:%M")
        if r["last_seen"] is not None:
            actual_arr = datetime.fromtimestamp(r["last_seen"], timezone.utc).strftime("%H:%M")
        has_actual = r["first_seen"] is not None and r["expected_duration_min"] is not None
        if r["is_delayed"] is not None:
            status = "Delayed" if r["is_delayed"] else "On time"
        elif not has_actual:
            status = "No data"
        else:
            status = "On time"
        actuals.append({
            "callsign": r["callsign"],
            "route": f"{r['origin']}→{r['destination']}",
            "aircraft_type": r["aircraft_type"],
            "scheduled_dep": scheduled_dep,
            "scheduled_arr": scheduled_arr,
            "actual_dep": actual_dep,
            "actual_arr": actual_arr,
            "expected_flight_min": r["expected_duration_min"],
            "actual_flight_min": r["actual_duration_min"],
            "deviation_min": round(r["deviation_min"], 1) if r["deviation_min"] is not None else None,
            "status": status,
            "has_actual": has_actual,
            "predicted_prob": r["delay_probability"],
            "predicted_delay_min": r["predicted_delay_min"],
            "predicted_risk": r["predicted_risk"],
            "actual_delay_min": r["actual_delay_min"],
            "actual_is_delayed": r["actual_is_delayed"],
        })
    return actuals


def update_daily_data(
    days_back: int = 1,
    db_path: Optional[Path] = None,
    retrain: bool = True,
) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    result = seed_historical_data(days=days_back, db_path=path)

    labels_added = 0
    if retrain:
        try:
            labels_added = compute_delay_labels(path)
        except Exception:
            pass

    model_result = {"status": "skipped"}
    if retrain:
        try:
            from ml_engine.delay_predictor import retrain_if_stale
            model_result = retrain_if_stale(max_age_hours=12, db_path=path)
        except Exception:
            pass

    return {
        "seed": result,
        "labels_added": labels_added,
        "model": model_result,
    }


_INDIAN_ICAO_PREFIXES = ("VO", "VA", "VI", "VE", "VY")


def sync_opensky_flights_to_db(
    csv_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from data.flights_db import insert_flight, get_flight, get_crew_for_flight, get_flights
    from data.flights_db import unassign_crew_from_flight, delete_flight
    from data.models import Flight, FlightStatus
    from data.staff_manager import auto_assign_flight, REQUIRED_CREW

    path = db_path or DEFAULT_DB_PATH
    today = datetime.now()
    schedule = get_flight_schedule(limit=SCHEDULE_LIMIT, db_path=path)
    if not schedule:
        return {"inserted": 0, "assigned": 0, "message": "No flights in schedule."}

    if csv_path is None:
        csv_path = str(Path(__file__).parent.parent / "crew_standby_list.csv")

    reconciled = 0
    schedule_ids = {f["callsign"] for f in schedule}
    for existing in get_flights(db_path=path):
        if existing.flight_id in schedule_ids:
            continue
        for crew in get_crew_for_flight(existing.flight_id, path):
            unassign_crew_from_flight(crew["crew_id"], existing.flight_id, path)
        delete_flight(existing.flight_id, path)
        reconciled += 1

    inserted = 0
    assigned = 0
    skipped = 0
    filled = 0

    for f in schedule:
        fid = f["callsign"]
        existing = get_flight(fid, path)
        if existing:
            existing_crew = get_crew_for_flight(fid, path)
            if len(existing_crew) >= sum(REQUIRED_CREW.values()):
                skipped += 1
            else:
                result = auto_assign_flight(fid, csv_path, path)
                if result.get("assigned_count", 0) > 0:
                    filled += 1
            continue

        dep_hour = f["avg_departure_hour"]
        dep_minute = f["avg_departure_minute"]
        duration = f["avg_duration_min"]
        std = today.replace(hour=dep_hour, minute=dep_minute, second=0, microsecond=0)

        dest = f["destination"]
        is_intl = not any(dest.startswith(p) for p in _INDIAN_ICAO_PREFIXES)

        flight = Flight(
            flight_id=fid,
            origin=f["origin"],
            destination=dest,
            std=std,
            aircraft_type="B737",
            gate="",
            terminal="",
            pax_count=0,
            flight_duration_min=duration,
            is_international=is_intl,
        )
        insert_flight(flight, path)
        inserted += 1

        result = auto_assign_flight(fid, csv_path, path)
        if result.get("assigned_count", 0) > 0:
            assigned += 1

    return {
        "inserted": inserted,
        "filled": filled,
        "skipped": skipped,
        "assigned": assigned,
        "reconciled": reconciled,
        "total": len(schedule),
        "message": f"Inserted {inserted} flights, assigned crew to {assigned}, backfilled {filled} existing flights, skipped {skipped} (already full), removed {reconciled} stale flights.",
    }


def log_predictions(
    today_flights: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
    target_date: Optional[str] = None,
) -> int:
    path = db_path or DEFAULT_DB_PATH

    def _write():
        conn = _connect_write(path)
        try:
            today_str = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            now_iso = datetime.now(timezone.utc).isoformat()
            count = 0
            for f in today_flights:
                pred = f.get("prediction", {})
                if not pred:
                    continue
                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO prediction_log
                        (callsign, origin, destination, date, predicted_at,
                         delay_probability, expected_delay_min, risk_level)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            f["callsign"], f["origin"], f["destination"], today_str, now_iso,
                            pred["delay_probability"], pred["expected_delay_min"], pred["risk_level"],
                        ),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            return count
        finally:
            conn.close()

    return _retry_on_lock(_write)


def backfill_predictions(db_path: Optional[Path] = None) -> int:
    path = db_path or DEFAULT_DB_PATH
    init_opensky_tables(db_path)
    conn = _connect(path)
    try:
        dates = conn.execute(
            "SELECT DISTINCT date FROM delay_labels WHERE date IS NOT NULL ORDER BY date"
        ).fetchall()
        existing = conn.execute(
            "SELECT DISTINCT date FROM prediction_log"
        ).fetchall()
        existing_dates = {r["date"] for r in existing}
    finally:
        conn.close()

    total = 0
    for row in dates:
        d = row["date"]
        if not d or d in existing_dates:
            continue
        try:
            schedule = get_schedule_for_date(d, db_path=path)
        except Exception:
            continue
        if not schedule:
            continue
        log_predictions(schedule, db_path=path, target_date=d)
        total += len(schedule)
    return total


def update_actuals(db_path: Optional[Path] = None) -> int:
    path = db_path or DEFAULT_DB_PATH

    def _write():
        conn = _connect_write(path)
        try:
            rows = conn.execute(
                """SELECT id, callsign, origin, destination, date
                   FROM prediction_log
                   WHERE actual_is_delayed IS NULL"""
            ).fetchall()
            count = 0
            for r in rows:
                actuals = conn.execute(
                    """SELECT deviation_min, is_delayed
                       FROM delay_labels
                       WHERE flight_id = ? AND origin = ? AND destination = ? AND date = ?
                       LIMIT 1""",
                    (r["callsign"], r["origin"], r["destination"], r["date"]),
                ).fetchall()
                if actuals:
                    a = actuals[0]
                    actual_delay = a["deviation_min"] if a["deviation_min"] is not None else 0
                    actual_delayed = a["is_delayed"] if a["is_delayed"] is not None else 0
                    conn.execute(
                        """UPDATE prediction_log
                           SET actual_delay_min = ?, actual_is_delayed = ?, actual_recorded_at = ?
                           WHERE id = ?""",
                        (round(actual_delay, 1), actual_delayed, datetime.now(timezone.utc).isoformat(), r["id"]),
                    )
                    count += 1
            conn.commit()
            return count
        finally:
            conn.close()

    return _retry_on_lock(_write)


def get_prediction_audit(days: int = 14, db_path: Optional[Path] = None) -> pd.DataFrame:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return pd.DataFrame()
    conn = _connect(path)
    try:
        df = pd.read_sql_query(
            """SELECT callsign, origin, destination, date, predicted_at,
                      delay_probability, expected_delay_min, risk_level,
                      actual_delay_min, actual_is_delayed
               FROM prediction_log
               WHERE actual_is_delayed IS NOT NULL
                 AND date >= date('now', ?)
               ORDER BY date DESC, delay_probability DESC""",
            conn,
            params=(f"-{days} days",),
        )
    except sqlite3.OperationalError:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def compute_audit_metrics(audit_df: pd.DataFrame) -> Dict[str, Any]:
    if audit_df.empty:
        return {"total": 0, "message": "No audit data available."}

    total = len(audit_df)
    predicted_high_med = audit_df["risk_level"].isin(["High", "Medium"])
    predicted_low = audit_df["risk_level"] == "Low"
    actually_delayed = audit_df["actual_is_delayed"] == 1
    actually_not_delayed = audit_df["actual_is_delayed"] == 0

    tp = int((predicted_high_med & actually_delayed).sum())
    fp = int((predicted_high_med & actually_not_delayed).sum())
    tn = int((predicted_low & actually_not_delayed).sum())
    fn = int((predicted_low & actually_delayed).sum())

    accuracy = round((tp + tn) / total * 100, 1) if total > 0 else 0
    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
    recall = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
    f1 = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0

    mask = audit_df["actual_delay_min"].notna()
    mae = round(audit_df.loc[mask, "expected_delay_min"].sub(audit_df.loc[mask, "actual_delay_min"].abs()).abs().mean(), 1) if mask.any() else 0
    mae_simple = round((audit_df.loc[mask, "expected_delay_min"] - audit_df.loc[mask, "actual_delay_min"].abs()).abs().mean(), 1) if mask.any() else 0

    brier = 0.0
    if total > 0:
        probs = audit_df["delay_probability"].clip(0, 1)
        actuals = audit_df["actual_is_delayed"].astype(float)
        brier = round(((probs - actuals) ** 2).mean(), 4)

    calibration = []
    for lo in range(0, 100, 10):
        hi = lo + 10
        bucket = audit_df[(audit_df["delay_probability"] * 100 >= lo) & (audit_df["delay_probability"] * 100 < hi)]
        if len(bucket) > 0:
            calibration.append({
                "bucket": f"{lo}-{hi}%",
                "predicted_avg": round(bucket["delay_probability"].mean() * 100, 1),
                "actual_rate": round(bucket["actual_is_delayed"].mean() * 100, 1),
                "count": len(bucket),
            })

    risk_dist = []
    for level in ["High", "Medium", "Low"]:
        sub = audit_df[audit_df["risk_level"] == level]
        if len(sub) > 0:
            risk_dist.append({
                "risk_level": level,
                "count": len(sub),
                "actual_delay_rate": round(sub["actual_is_delayed"].mean() * 100, 1),
                "avg_predicted_prob": round(sub["delay_probability"].mean() * 100, 1),
                "avg_actual_delay_min": round(sub["actual_delay_min"].abs().mean(), 1) if sub["actual_delay_min"].notna().any() else 0,
            })

    audit_df = audit_df.copy()
    audit_df["predicted_delayed"] = audit_df["risk_level"].isin(["High", "Medium"])
    audit_df["match"] = audit_df.apply(
        lambda r: "Correct" if r["predicted_delayed"] == bool(r["actual_is_delayed"])
        else ("Miss" if r["actual_is_delayed"] == 1 else "False Alarm"),
        axis=1,
    )

    return {
        "total": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae_simple,
        "brier_score": brier,
        "calibration": calibration,
        "risk_distribution": risk_dist,
        "predictions": audit_df[["date", "callsign", "origin", "destination", "delay_probability",
                                  "risk_level", "actual_delay_min", "actual_is_delayed", "match"]].to_dict("records"),
    }
