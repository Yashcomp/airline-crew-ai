from __future__ import annotations

import csv
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path(__file__).parent / "flights.db"
_CALLSIGN_MAP_PATH = Path(__file__).parent / "callsign_map.csv"
DELAY_THRESHOLD_MIN = 15


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_tracked_flights(
    db_path: Optional[Path] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []

    callsign_map = _load_callsign_map()
    conn = _connect(path)
    try:
        rows = conn.execute("""
            SELECT
                callsign,
                icao24,
                origin_airport,
                destination_airport,
                aircraft_type,
                ROUND(AVG(
                    CAST((CAST(first_seen AS INTEGER) % 86400) / 3600 AS REAL)
                ), 1) as avg_hour,
                ROUND(AVG(
                    CAST(((CAST(first_seen AS INTEGER) % 86400) % 3600) / 60.0 AS REAL)
                ), 0) as avg_minute,
                ROUND(AVG(duration_min), 0) as avg_duration,
                COUNT(*) as flight_count,
                COUNT(DISTINCT date) as days_active
            FROM opensky_flights
            WHERE callsign IS NOT NULL AND callsign != ''
              AND first_seen IS NOT NULL
              AND (origin_airport = 'VOBL' OR destination_airport = 'VOBL')
            GROUP BY callsign
            HAVING flight_count >= 2
            ORDER BY flight_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    flights = []
    for r in rows:
        callsign = r["callsign"]
        cs_upper = callsign.upper()

        map_entry = callsign_map.get(cs_upper, {})
        flight_id = map_entry.get("flight_id") or callsign

        h = r["avg_hour"]
        m = r["avg_minute"]
        hour = int(h) if h else 12
        minute = int(m) if m else 0

        entry = {
            "callsign": callsign,
            "flight_id": flight_id,
            "origin": r["origin_airport"] or "VOBL",
            "destination": r["destination_airport"] or "Unknown",
            "aircraft_type": r["aircraft_type"] or map_entry.get("aircraft_type", ""),
            "scheduled_departure": f"{hour:02d}:{minute:02d}",
            "scheduled_dep_hour": hour,
            "scheduled_dep_minute": minute,
            "avg_duration_min": int(r["avg_duration"] or 0),
            "icao24": r["icao24"] or _callsign_to_icao24(callsign),
            "flight_count": r["flight_count"],
            "days_active": r["days_active"],
        }
        flights.append(entry)

    return flights


def detect_delays(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    tracked = load_tracked_flights(path)
    if not tracked:
        return []

    now = int(time.time())
    two_hours_ago = now - 7200

    all_callsigns = [f["callsign"] for f in tracked]
    all_icao24s = [f["icao24"] for f in tracked if f.get("icao24")]
    states_map = _batch_query_states(all_callsigns, all_icao24s, two_hours_ago, path)

    results = []
    for flight in tracked:
        cs = flight["callsign"]
        icao = flight.get("icao24", "")
        states = states_map.get(cs, []) or states_map.get(icao, [])
        departure_event = _detect_departure_event(states)

        status = "Unknown"
        actual_departure = None
        delay_minutes = None

        if departure_event:
            status = "Departed"
            actual_departure = departure_event["timestamp"]
            scheduled_ts = _scheduled_to_timestamp(
                flight["scheduled_dep_hour"], flight["scheduled_dep_minute"]
            )
            delay_minutes = round((actual_departure - scheduled_ts) / 60)
            if delay_minutes < -10:
                status = "Early"
        elif states:
            latest = states[-1]
            if latest["on_ground"]:
                if latest.get("velocity_ms", 0) > 2:
                    status = "Taxiing"
                else:
                    status = "At Gate"
            else:
                alt = latest.get("altitude_m") or 0
                if alt > 1000:
                    status = "In Air"
                else:
                    status = "Climbing"
        else:
            status = "No Data"

        results.append({
            **flight,
            "status": status,
            "actual_departure": actual_departure,
            "actual_departure_str": (
                datetime.fromtimestamp(actual_departure, tz=timezone.utc).strftime("%H:%M")
                if actual_departure else None
            ),
            "delay_minutes": delay_minutes,
            "state_count": len(states),
            "last_seen": states[-1]["timestamp"] if states else None,
        })

    return results


def detect_round_trip_delays(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    flight_results = detect_delays(path)

    icao_groups: Dict[str, List[Dict]] = {}
    for f in flight_results:
        icao = f.get("icao24", "")
        if icao:
            icao_groups.setdefault(icao, []).append(f)

    enhanced = []
    for icao, flights in icao_groups.items():
        if len(flights) == 1:
            enhanced.append(flights[0])
            continue

        outbound = None
        inbound = None
        for f in flights:
            dep = f["scheduled_dep_hour"]
            if dep < 14:
                outbound = f
            else:
                inbound = f

        if not outbound and not inbound:
            enhanced.extend(flights)
            continue

        if outbound and not inbound:
            enhanced.append(outbound)
        elif inbound and not outbound:
            enhanced.append(inbound)
        elif outbound and inbound:
            inbound_delayed = (
                inbound.get("delay_minutes") is not None
                and (inbound["delay_minutes"] or 0) > DELAY_THRESHOLD_MIN
            )
            if inbound_delayed:
                inbound["at_risk"] = True
                inbound["risk_reason"] = (
                    f"Inbound {inbound['callsign']} delayed "
                    f"+{inbound['delay_minutes']}min — may affect outbound "
                    f"{outbound['callsign']}"
                )
            enhanced.append(inbound)
            enhanced.append(outbound)

    return enhanced


def get_flight_statuses(db_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    flights = detect_delays(db_path)
    return {f["callsign"]: f for f in flights}


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


def _callsign_to_icao24(callsign: str) -> str:
    cs = callsign.upper()
    if cs.startswith("AIC"):
        return f"800{cs[3:]}" if len(cs) <= 6 else cs[:8]
    if cs.startswith("IGO"):
        return f"89{cs[3:]}" if len(cs) <= 6 else cs[:8]
    return cs[:8]


def _batch_query_states(
    callsigns: List[str],
    icao24s: List[str],
    since_timestamp: int,
    db_path: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    result_map: Dict[str, List[Dict[str, Any]]] = {}
    if not callsigns and not icao24s:
        return result_map

    conn = _connect(db_path)
    try:
        upper_callsigns = [c.upper() for c in callsigns if c]
        if upper_callsigns:
            placeholders = ",".join("?" for _ in upper_callsigns)
            rows = conn.execute(
                f"""SELECT timestamp, altitude_m, velocity_ms, on_ground, vertical_rate,
                           latitude, longitude, callsign, icao24
                    FROM opensky_states
                    WHERE UPPER(callsign) IN ({placeholders}) AND timestamp > ?
                    ORDER BY timestamp ASC""",
                upper_callsigns + [since_timestamp],
            ).fetchall()
            for r in rows:
                cs = (r["callsign"] or "").upper()
                if cs:
                    result_map.setdefault(cs, []).append(dict(r))
                icao = r["icao24"] or ""
                if icao:
                    result_map.setdefault(icao, []).append(dict(r))

        if icao24s:
            unique_icao = list(set(icao24s))
            placeholders = ",".join("?" for _ in unique_icao)
            rows = conn.execute(
                f"""SELECT timestamp, altitude_m, velocity_ms, on_ground, vertical_rate,
                           latitude, longitude, callsign, icao24
                    FROM opensky_states
                    WHERE icao24 IN ({placeholders}) AND timestamp > ?
                    ORDER BY timestamp ASC""",
                unique_icao + [since_timestamp],
            ).fetchall()
            for r in rows:
                icao = r["icao24"] or ""
                if icao:
                    existing = result_map.get(icao, [])
                    existing_ts = {s["timestamp"] for s in existing}
                    if r["timestamp"] not in existing_ts:
                        existing.append(dict(r))
                    result_map[icao] = existing
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()

    return result_map


def _detect_departure_event(
    states: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    for i in range(1, len(states)):
        prev = states[i - 1]
        curr = states[i]
        if prev["on_ground"] and not curr["on_ground"]:
            alt = curr.get("altitude_m") or 0
            if alt > 30:
                return {
                    "timestamp": curr["timestamp"],
                    "altitude_m": alt,
                    "velocity_ms": curr.get("velocity_ms", 0),
                }
    return None


def _scheduled_to_timestamp(hour: int, minute: int) -> int:
    today = datetime.now(timezone.utc).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return int(today.timestamp())


def store_realtime_delays(
    delays: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> None:
    path = db_path or DEFAULT_DB_PATH
    _init_realtime_delays_table(path)
    conn = _connect(path)
    conn.execute("DELETE FROM realtime_delays")
    for d in delays:
        try:
            conn.execute(
                """INSERT INTO realtime_delays
                (flight_id, callsign, status, scheduled_departure,
                 actual_departure, delay_minutes, last_seen, detected_at,
                 origin, destination, at_risk, risk_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d.get("flight_id"),
                    d.get("callsign"),
                    d.get("status"),
                    d.get("scheduled_departure"),
                    d.get("actual_departure"),
                    d.get("delay_minutes"),
                    d.get("last_seen"),
                    int(time.time()),
                    d.get("origin"),
                    d.get("destination"),
                    1 if d.get("at_risk") else 0,
                    d.get("risk_reason"),
                ),
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    conn.close()


def get_realtime_delays(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    _init_realtime_delays_table(path)
    if not path.exists():
        return []
    conn = _connect(path)
    try:
        rows = conn.execute(
            """SELECT * FROM realtime_delays ORDER BY
               CASE status
                 WHEN 'Departed' THEN 1
                 WHEN 'Early' THEN 1
                 WHEN 'Climbing' THEN 2
                 WHEN 'In Air' THEN 3
                 WHEN 'Taxiing' THEN 4
                 WHEN 'At Gate' THEN 5
                 ELSE 6
               END, delay_minutes DESC"""
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _init_realtime_delays_table(db_path: Path) -> None:
    conn = _connect(db_path)
    conn.execute("""
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
        )
    """)
    conn.commit()
    conn.close()
