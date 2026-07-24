from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.models import Flight, FlightStatus

DEFAULT_DB_PATH = Path(__file__).parent / "flights.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Optional[Path] = None) -> Path:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            flight_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            std TEXT NOT NULL,
            aircraft_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'scheduled',
            gate TEXT,
            terminal TEXT,
            pax_count INTEGER DEFAULT 0,
            turnaround_min INTEGER DEFAULT 45,
            flight_duration_min INTEGER DEFAULT 120,
            is_international INTEGER DEFAULT 0,
            disruption_reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crew_assignments (
            assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            crew_id TEXT NOT NULL,
            flight_id TEXT NOT NULL,
            role TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'assigned',
            UNIQUE(crew_id, flight_id)
        )
    """)
    conn.commit()
    conn.close()
    return path


def _row_to_flight(row: sqlite3.Row) -> Flight:
    return Flight(
        flight_id=row["flight_id"],
        origin=row["origin"],
        destination=row["destination"],
        std=datetime.fromisoformat(row["std"]),
        aircraft_type=row["aircraft_type"],
        status=FlightStatus(row["status"]),
        gate=row["gate"],
        terminal=row["terminal"],
        pax_count=row["pax_count"] or 0,
        turnaround_min=row["turnaround_min"] or 45,
        flight_duration_min=row["flight_duration_min"] or 120,
        is_international=bool(row["is_international"]),
        disruption_reason=row["disruption_reason"],
    )


def insert_flight(flight: Flight, db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("""
        INSERT OR REPLACE INTO flights
        (flight_id, origin, destination, std, aircraft_type, status,
         gate, terminal, pax_count, turnaround_min, flight_duration_min,
         is_international, disruption_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        flight.flight_id, flight.origin, flight.destination,
        flight.std.isoformat(), flight.aircraft_type, flight.status.value,
        flight.gate, flight.terminal, flight.pax_count,
        flight.turnaround_min, flight.flight_duration_min,
        int(flight.is_international), flight.disruption_reason,
    ))
    conn.commit()
    conn.close()


def insert_flights(flights: List[Flight], db_path: Optional[Path] = None) -> None:
    for f in flights:
        insert_flight(f, db_path)


def get_flights(
    db_path: Optional[Path] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[datetime] = None,
    status: Optional[str] = None,
    aircraft_type: Optional[str] = None,
) -> List[Flight]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    query = "SELECT * FROM flights WHERE 1=1"
    params: List[Any] = []

    if origin:
        query += " AND origin = ?"
        params.append(origin.upper())
    if destination:
        query += " AND destination = ?"
        params.append(destination.upper())
    if date:
        day_start = date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        day_end = (date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
        query += " AND std >= ? AND std < ?"
        params.extend([day_start, day_end])
    if status:
        query += " AND status = ?"
        params.append(status)
    if aircraft_type:
        query += " AND aircraft_type = ?"
        params.append(aircraft_type.upper())

    query += " ORDER BY std ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_flight(r) for r in rows]


def get_flight(flight_id: str, db_path: Optional[Path] = None) -> Optional[Flight]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return None
    conn = _connect(path)
    row = conn.execute("SELECT * FROM flights WHERE flight_id = ?", (flight_id,)).fetchone()
    conn.close()
    return _row_to_flight(row) if row else None


def update_flight_status(
    flight_id: str,
    status: str,
    disruption_reason: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    if disruption_reason:
        conn.execute(
            "UPDATE flights SET status = ?, disruption_reason = ? WHERE flight_id = ?",
            (status, disruption_reason, flight_id),
        )
    else:
        conn.execute(
            "UPDATE flights SET status = ? WHERE flight_id = ?",
            (status, flight_id),
        )
    conn.commit()
    conn.close()


def delete_flight(flight_id: str, db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute("DELETE FROM flights WHERE flight_id = ?", (flight_id,))
    conn.commit()
    conn.close()


def get_disrupted_flights(db_path: Optional[Path] = None) -> List[Flight]:
    return get_flights(db_path=db_path, status="delayed") + get_flights(db_path=db_path, status="cancelled")


def get_upcoming_flights(hours_ahead: int = 6, origin: Optional[str] = None, db_path: Optional[Path] = None) -> List[Flight]:
    now = datetime.now()
    cutoff = now + timedelta(hours=hours_ahead)
    flights = get_flights(db_path=db_path, origin=origin)
    def _naive(dt):
        return dt.replace(tzinfo=None) if dt and dt.tzinfo else dt
    return [f for f in flights if now <= _naive(f.std) <= cutoff and f.status not in (FlightStatus.DEPARTED, FlightStatus.LANDED, FlightStatus.CANCELLED)]


def get_flight_stats(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return {"total": 0, "by_status": {}, "by_aircraft": {}}
    conn = _connect(path)
    total = conn.execute("SELECT COUNT(*) FROM flights").fetchone()[0]
    status_rows = conn.execute("SELECT status, COUNT(*) as cnt FROM flights GROUP BY status").fetchall()
    aircraft_rows = conn.execute("SELECT aircraft_type, COUNT(*) as cnt FROM flights GROUP BY aircraft_type").fetchall()
    conn.close()
    return {
        "total": total,
        "by_status": {r["status"]: r["cnt"] for r in status_rows},
        "by_aircraft": {r["aircraft_type"]: r["cnt"] for r in aircraft_rows},
    }


def clear_db(db_path: Optional[Path] = None) -> None:
    path = db_path or DEFAULT_DB_PATH
    if path.exists():
        conn = _connect(path)
        conn.execute("DELETE FROM flights")
        conn.execute("DELETE FROM crew_assignments")
        conn.commit()
        conn.close()


def assign_crew_to_flight(
    crew_id: str,
    flight_id: str,
    role: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT INTO crew_assignments (crew_id, flight_id, role, assigned_at, status) VALUES (?, ?, ?, ?, ?)",
            (crew_id.upper(), flight_id.upper(), role, datetime.now().isoformat(), "assigned"),
        )
        conn.commit()
        return {"status": "success", "crew_id": crew_id.upper(), "flight_id": flight_id.upper()}
    except sqlite3.IntegrityError:
        return {"status": "error", "message": f"{crew_id} is already assigned to {flight_id}"}
    finally:
        conn.close()


def unassign_crew_from_flight(
    crew_id: str,
    flight_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB_PATH
    conn = _connect(path)
    conn.execute(
        "DELETE FROM crew_assignments WHERE crew_id = ? AND flight_id = ?",
        (crew_id.upper(), flight_id.upper()),
    )
    conn.commit()
    conn.close()
    return {"status": "success", "message": f"{crew_id} unassigned from {flight_id}"}


def get_crew_for_flight(
    flight_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    rows = conn.execute(
        "SELECT * FROM crew_assignments WHERE flight_id = ? AND status = 'assigned'",
        (flight_id.upper(),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_flights_for_crew(
    crew_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    rows = conn.execute(
        "SELECT * FROM crew_assignments WHERE crew_id = ? AND status = 'assigned'",
        (crew_id.upper(),),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_crew_assigned(
    crew_id: str,
    db_path: Optional[Path] = None,
) -> bool:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return False
    conn = _connect(path)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM crew_assignments WHERE crew_id = ? AND status = 'assigned'",
        (crew_id.upper(),),
    ).fetchone()
    conn.close()
    return row["cnt"] > 0


def get_all_assignments(
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    path = db_path or DEFAULT_DB_PATH
    if not path.exists():
        return []
    conn = _connect(path)
    rows = conn.execute(
        "SELECT * FROM crew_assignments WHERE status = 'assigned' ORDER BY assigned_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_standby_crew(
    csv_path: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    from data.crew_loader import load_crew
    crew = load_crew(csv_path)
    path = db_path or DEFAULT_DB_PATH    
    standby = []
    for member in crew:
        if not is_crew_assigned(member.crew_id, path):
            standby.append({
                "crew_id": member.crew_id,
                "name": member.name,
                "role": member.role.value,
                "rest_status": member.rest_status,
                "base_airport": member.base_airport,
                "current_duty_hours": member.current_duty_hours,
                "rolling_7_day_hours": member.rolling_7_day_hours,
            })
    return standby
