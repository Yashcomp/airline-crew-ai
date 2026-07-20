from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.models import Role

DATASET_DIR = Path(__file__).parent / "airport-operations-dataset"

# ---------------------------------------------------------------------------
# Column maps derived from raw CSV inspection (0-indexed)
# ---------------------------------------------------------------------------
FLIGHTS_COLS = [
    "flight_id", "airline", "iata_code", "origin", "destination",
    "std", "actual_dep", "arrival", "sta", "aircraft_type", "registration",
    "capacity", "booked_pax", "status", "delay_min", "delay_reason",
    "terminal", "gate", "international", "distance", "fuel_cost",
    "turbulence_index", "boarding_time", "is_heavy", "turbulence_category",
    "seat_occupancy", "connecting_pax", "time_of_day", "day_of_week",
    "is_weekend", "season", "route_type",
]

STAFF_SHIFT_COLS = [
    "staff_id", "name", "department", "role_raw", "shift_date",
    "shift_start", "shift_end", "terminal", "gate", "assignment_id",
    "shift_hours", "overtime", "_blank", "valid_until", "language",
]

GATE_EVENT_COLS = [
    "event_id", "flight_id", "gate", "terminal", "event_type",
    "timestamp", "staff_id", "duration_min", "event_category", "escalated",
    "_blank", "timestamp2", "timestamp3", "timestamp4",
]

BAGGAGE_COLS = [
    "baggage_tag", "pnr_code", "flight_id", "passenger_id", "weight",
    "dimensions", "stage", "gate", "checkin_time", "loaded_time",
    "pieces", "status", "delay", "delay_mins", "assigned_staff",
    "assigned_at", "is_oversized", "_blank",
]

PASSENGER_COLS = [
    "pnr_code", "ticket_no", "passenger_id", "first_name", "last_name",
    "nationality", "dob", "gender", "seat", "seat_class", "flight_id",
    "booking_time", "boarding_time", "gate", "baggage_count",
    "_b1", "_b2", "_b3", "_b4", "contact_email", "contact_phone",
    "_b5", "_b6", "upgrade", "loyalty_score", "is_connecting",
    "fare_class", "fare_amount", "age_category",
]

SECURITY_COLS = [
    "screen_id", "passenger_id", "ticket_code", "screen_type",
    "queue_start", "queue_end", "clearance_time", "result",
    "alarm_type", "secondary", "staff_id", "xray_machine",
    "processing_time", "has_alarm", "has_secondary", "shift_id",
    "pass_per_hour", "capacity", "utilization", "escalated",
]

MAINTENANCE_COLS = [
    "work_order", "aircraft_reg", "flight_id", "task_type", "staff_id",
    "assigned_at", "resolved_at", "severity", "duration_hrs", "defect",
    "component", "impact_level", "resolved_by", "airworthy", "escalated",
    "_blank",
]

RETAIL_COLS = [
    "txn_id", "staff_id", "store_name", "store_type", "passenger_id",
    "flight_id", "txn_time", "product", "quantity", "amount", "discount",
    "payment_method", "currency", "_blank", "terminal", "location",
    "is_connecting",
]

# Role inference from staff_id prefix
_ROLE_PREFIX_MAP = {
    "SEC": Role.SECURITY,
    "GH": Role.GROUND_HANDLING,
    "MTC": Role.MAINTENANCE,
    "OPS": Role.OPERATIONS,
    "CC": Role.CABIN_CLEANING,
    "RET": Role.RETAIL,
}

_BLANK_NAMES = {"_blank", "_b1", "_b2", "_b3", "_b4", "_b5", "_b6"}


def infer_role(staff_id: str) -> Role:
    prefix = staff_id.split("-")[0] if "-" in staff_id else ""
    return _ROLE_PREFIX_MAP.get(prefix, Role.GROUND_STAFF)


def _safe_float(val: str, default: float = 0.0) -> float:
    if not val or not val.strip():
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _safe_int(val: str, default: int = 0) -> int:
    if not val or not val.strip():
        return default
    try:
        return int(float(val))
    except ValueError:
        return default


def _safe_bool(val: str) -> int:
    return 1 if val and val.strip().lower() in ("true", "1", "yes") else 0


def _read_csv_rows(csv_path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # skip header row (column indices)
            rows.append(row)
    return rows


def _map_row(row: List[str], col_map: List[str]) -> Dict[str, str]:
    result = {}
    for i, col_name in enumerate(col_map):
        if col_name in _BLANK_NAMES:
            continue
        if i < len(row):
            result[col_name] = row[i]
        else:
            result[col_name] = ""
    return result


# ---------------------------------------------------------------------------
# Table creation SQL
# ---------------------------------------------------------------------------
_TABLE_SQL = {
    "ops_flights": """
        CREATE TABLE IF NOT EXISTS ops_flights (
            flight_id TEXT PRIMARY KEY,
            airline TEXT,
            iata_code TEXT,
            origin TEXT,
            destination TEXT,
            std TEXT,
            actual_dep TEXT,
            arrival TEXT,
            sta TEXT,
            aircraft_type TEXT,
            registration TEXT,
            capacity INTEGER DEFAULT 0,
            booked_pax INTEGER DEFAULT 0,
            status TEXT,
            delay_min INTEGER DEFAULT 0,
            delay_reason TEXT,
            terminal TEXT,
            gate TEXT,
            international INTEGER DEFAULT 0,
            distance REAL DEFAULT 0,
            fuel_cost REAL DEFAULT 0,
            turbulence_index REAL DEFAULT 0,
            boarding_time TEXT,
            is_heavy INTEGER DEFAULT 0,
            turbulence_category TEXT,
            seat_occupancy REAL DEFAULT 0,
            connecting_pax INTEGER DEFAULT 0,
            time_of_day TEXT,
            day_of_week TEXT,
            is_weekend INTEGER DEFAULT 0,
            season TEXT,
            route_type TEXT
        )
    """,
    "ops_staff": """
        CREATE TABLE IF NOT EXISTS ops_staff (
            staff_id TEXT PRIMARY KEY,
            name TEXT,
            department TEXT,
            role TEXT,
            terminal TEXT,
            valid_until TEXT
        )
    """,
    "ops_shifts": """
        CREATE TABLE IF NOT EXISTS ops_shifts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            staff_id TEXT,
            shift_date TEXT,
            shift_start TEXT,
            shift_end TEXT,
            terminal TEXT,
            gate TEXT,
            shift_hours REAL DEFAULT 0,
            overtime INTEGER DEFAULT 0
        )
    """,
    "ops_gate_events": """
        CREATE TABLE IF NOT EXISTS ops_gate_events (
            event_id TEXT PRIMARY KEY,
            flight_id TEXT,
            gate TEXT,
            terminal TEXT,
            event_type TEXT,
            timestamp TEXT,
            staff_id TEXT,
            duration_min INTEGER DEFAULT 0,
            event_category TEXT,
            escalated INTEGER DEFAULT 0
        )
    """,
    "ops_passengers": """
        CREATE TABLE IF NOT EXISTS ops_passengers (
            pnr_code TEXT,
            ticket_no TEXT,
            passenger_id TEXT PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            nationality TEXT,
            dob TEXT,
            gender TEXT,
            seat TEXT,
            seat_class TEXT,
            flight_id TEXT,
            booking_time TEXT,
            boarding_time TEXT,
            gate TEXT,
            baggage_count INTEGER DEFAULT 0,
            upgrade INTEGER DEFAULT 0,
            loyalty_score REAL DEFAULT 0,
            is_connecting INTEGER DEFAULT 0,
            fare_class TEXT,
            fare_amount REAL DEFAULT 0,
            age_category TEXT
        )
    """,
    "ops_baggage": """
        CREATE TABLE IF NOT EXISTS ops_baggage (
            baggage_tag TEXT PRIMARY KEY,
            pnr_code TEXT,
            flight_id TEXT,
            passenger_id TEXT,
            weight REAL DEFAULT 0,
            dimensions TEXT,
            stage TEXT,
            gate TEXT,
            checkin_time TEXT,
            loaded_time TEXT,
            pieces INTEGER DEFAULT 0,
            status TEXT,
            delay INTEGER DEFAULT 0,
            delay_mins INTEGER DEFAULT 0,
            assigned_staff TEXT,
            assigned_at TEXT,
            is_oversized INTEGER DEFAULT 0
        )
    """,
    "ops_security": """
        CREATE TABLE IF NOT EXISTS ops_security (
            screen_id TEXT PRIMARY KEY,
            passenger_id TEXT,
            ticket_code TEXT,
            screen_type INTEGER DEFAULT 0,
            queue_start TEXT,
            queue_end TEXT,
            clearance_time TEXT,
            result TEXT,
            alarm_type TEXT,
            secondary INTEGER DEFAULT 0,
            staff_id TEXT,
            xray_machine TEXT,
            processing_time REAL DEFAULT 0,
            has_alarm INTEGER DEFAULT 0,
            has_secondary INTEGER DEFAULT 0,
            shift_id TEXT,
            pass_per_hour INTEGER DEFAULT 0,
            capacity INTEGER DEFAULT 0,
            utilization REAL DEFAULT 0,
            escalated INTEGER DEFAULT 0
        )
    """,
    "ops_maintenance": """
        CREATE TABLE IF NOT EXISTS ops_maintenance (
            work_order TEXT PRIMARY KEY,
            aircraft_reg TEXT,
            flight_id TEXT,
            task_type TEXT,
            staff_id TEXT,
            assigned_at TEXT,
            resolved_at TEXT,
            severity INTEGER DEFAULT 0,
            duration_hrs REAL DEFAULT 0,
            defect TEXT,
            component TEXT,
            impact_level INTEGER DEFAULT 0,
            resolved_by TEXT,
            airworthy INTEGER DEFAULT 0,
            escalated INTEGER DEFAULT 0
        )
    """,
    "ops_retail": """
        CREATE TABLE IF NOT EXISTS ops_retail (
            txn_id TEXT PRIMARY KEY,
            staff_id TEXT,
            store_name TEXT,
            store_type TEXT,
            passenger_id TEXT,
            flight_id TEXT,
            txn_time TEXT,
            product TEXT,
            quantity INTEGER DEFAULT 0,
            amount REAL DEFAULT 0,
            discount REAL DEFAULT 0,
            payment_method TEXT,
            currency TEXT,
            terminal TEXT,
            location TEXT,
            is_connecting INTEGER DEFAULT 0
        )
    """,
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_flights(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "flights.csv")
    count = 0
    for row in rows:
        m = _map_row(row, FLIGHTS_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_flights VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["flight_id"], m["airline"], m["iata_code"],
                m["origin"], m["destination"], m["std"],
                m["actual_dep"], m["arrival"], m["sta"],
                m["aircraft_type"], m["registration"],
                _safe_int(m["capacity"]), _safe_int(m["booked_pax"]),
                m["status"], _safe_int(m["delay_min"]), m["delay_reason"],
                m["terminal"], m["gate"], _safe_bool(m["international"]),
                _safe_float(m["distance"]), _safe_float(m["fuel_cost"]),
                _safe_float(m["turbulence_index"]), m["boarding_time"],
                _safe_bool(m["is_heavy"]), m["turbulence_category"],
                _safe_float(m["seat_occupancy"]),
                _safe_int(m["connecting_pax"]),
                m["time_of_day"], m["day_of_week"],
                _safe_bool(m["is_weekend"]), m["season"], m["route_type"],
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_staff_and_shifts(conn: sqlite3.Connection) -> Tuple[int, int]:
    rows = _read_csv_rows(DATASET_DIR / "staff_shifts.csv")
    staff_seen: Dict[str, bool] = {}
    shift_count = 0
    for row in rows:
        m = _map_row(row, STAFF_SHIFT_COLS)
        sid = m["staff_id"]
        role = infer_role(sid)

        if sid not in staff_seen:
            conn.execute(
                "INSERT OR REPLACE INTO ops_staff VALUES (?,?,?,?,?,?)",
                (sid, m["name"], m["department"], role.value,
                 m["terminal"], m["valid_until"]),
            )
            staff_seen[sid] = True

        conn.execute(
            "INSERT INTO ops_shifts (staff_id,shift_date,shift_start,shift_end,terminal,gate,shift_hours,overtime) VALUES (?,?,?,?,?,?,?,?)",
            (
                sid, m["shift_date"], m["shift_start"], m["shift_end"],
                m["terminal"], m["gate"],
                _safe_float(m["shift_hours"]),
                _safe_bool(m["overtime"]),
            ),
        )
        shift_count += 1
    conn.commit()
    return len(staff_seen), shift_count


def _load_gate_events(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "gate_events.csv")
    count = 0
    for row in rows:
        m = _map_row(row, GATE_EVENT_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_gate_events VALUES
               (?,?,?,?,?,?,?,?,?,?)""",
            (
                m["event_id"], m["flight_id"], m["gate"], m["terminal"],
                m["event_type"], m["timestamp"], m["staff_id"],
                _safe_int(m["duration_min"]), m["event_category"],
                _safe_bool(m["escalated"]),
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_passengers(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "passengers.csv")
    count = 0
    for row in rows:
        m = _map_row(row, PASSENGER_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_passengers VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["pnr_code"], m["ticket_no"], m["passenger_id"],
                m["first_name"], m["last_name"], m["nationality"],
                m["dob"], m["gender"], m["seat"], m["seat_class"],
                m["flight_id"], m["booking_time"], m["boarding_time"],
                m["gate"], _safe_int(m["baggage_count"]),
                _safe_bool(m["upgrade"]),
                _safe_float(m["loyalty_score"]),
                _safe_bool(m["is_connecting"]),
                m["fare_class"], _safe_float(m["fare_amount"]),
                m["age_category"],
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_baggage(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "baggage.csv")
    count = 0
    for row in rows:
        m = _map_row(row, BAGGAGE_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_baggage VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["baggage_tag"], m["pnr_code"], m["flight_id"],
                m["passenger_id"], _safe_float(m["weight"]),
                m["dimensions"], m["stage"], m["gate"],
                m["checkin_time"], m["loaded_time"],
                _safe_int(m["pieces"]), m["status"],
                _safe_bool(m["delay"]), _safe_int(m["delay_mins"]),
                m["assigned_staff"], m["assigned_at"],
                _safe_bool(m["is_oversized"]),
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_security(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "security_screening.csv")
    count = 0
    for row in rows:
        m = _map_row(row, SECURITY_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_security VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["screen_id"], m["passenger_id"], m["ticket_code"],
                _safe_int(m["screen_type"]),
                m["queue_start"], m["queue_end"], m["clearance_time"],
                m["result"], m["alarm_type"],
                _safe_bool(m["secondary"]),
                m["staff_id"], m["xray_machine"],
                _safe_float(m["processing_time"]),
                _safe_bool(m["has_alarm"]),
                _safe_bool(m["has_secondary"]),
                m["shift_id"],
                _safe_int(m["pass_per_hour"]),
                _safe_int(m["capacity"]),
                _safe_float(m["utilization"]),
                _safe_bool(m["escalated"]),
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_maintenance(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "maintenance_logs.csv")
    count = 0
    for row in rows:
        m = _map_row(row, MAINTENANCE_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_maintenance VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["work_order"], m["aircraft_reg"], m["flight_id"],
                m["task_type"], m["staff_id"],
                m["assigned_at"], m["resolved_at"],
                _safe_int(m["severity"]), _safe_float(m["duration_hrs"]),
                m["defect"], m["component"],
                _safe_int(m["impact_level"]),
                m["resolved_by"],
                _safe_bool(m["airworthy"]),
                _safe_bool(m["escalated"]),
            ),
        )
        count += 1
    conn.commit()
    return count


def _load_retail(conn: sqlite3.Connection) -> int:
    rows = _read_csv_rows(DATASET_DIR / "retail_transactions.csv")
    count = 0
    for row in rows:
        m = _map_row(row, RETAIL_COLS)
        conn.execute(
            """INSERT OR REPLACE INTO ops_retail VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                m["txn_id"], m["staff_id"], m["store_name"],
                m["store_type"], m["passenger_id"], m["flight_id"],
                m["txn_time"], m["product"], _safe_int(m["quantity"]),
                _safe_float(m["amount"]), _safe_float(m["discount"]),
                m["payment_method"], m["currency"],
                m["terminal"], m["location"],
                _safe_bool(m["is_connecting"]),
            ),
        )
        count += 1
    conn.commit()
    return count


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def create_ops_tables(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    for sql in _TABLE_SQL.values():
        conn.execute(sql)
    conn.commit()
    conn.close()


def load_ops_dataset(
    db_path: Optional[Path] = None,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    global DATASET_DIR
    if dataset_dir:
        DATASET_DIR = dataset_dir

    path = db_path or (Path(__file__).parent / "flights.db")
    create_ops_tables(path)

    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")

    flights = _load_flights(conn)
    staff_count, shifts = _load_staff_and_shifts(conn)
    gate_events = _load_gate_events(conn)
    passengers = _load_passengers(conn)
    baggage = _load_baggage(conn)
    security = _load_security(conn)
    maintenance = _load_maintenance(conn)
    retail = _load_retail(conn)

    conn.close()

    return {
        "status": "success",
        "flights": flights,
        "staff": staff_count,
        "shifts": shifts,
        "gate_events": gate_events,
        "passengers": passengers,
        "baggage": baggage,
        "security": security,
        "maintenance": maintenance,
        "retail": retail,
        "total": flights + staff_count + shifts + gate_events + passengers + baggage + security + maintenance + retail,
    }


# ---------------------------------------------------------------------------
# Query helpers — join queries across tables
# ---------------------------------------------------------------------------

def get_flight_full_profile(flight_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or (Path(__file__).parent / "flights.db")
    if not path.exists():
        return {"error": "Database not found"}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    flight = conn.execute("SELECT * FROM ops_flights WHERE flight_id = ?", (flight_id,)).fetchone()
    if not flight:
        conn.close()
        return {"error": f"Flight {flight_id} not found"}

    result = dict(flight)

    result["passengers"] = [dict(r) for r in conn.execute(
        "SELECT * FROM ops_passengers WHERE flight_id = ?", (flight_id,)
    ).fetchall()]

    result["baggage"] = [dict(r) for r in conn.execute(
        "SELECT * FROM ops_baggage WHERE flight_id = ?", (flight_id,)
    ).fetchall()]

    result["gate_events"] = [dict(r) for r in conn.execute(
        "SELECT * FROM ops_gate_events WHERE flight_id = ?", (flight_id,)
    ).fetchall()]

    result["maintenance"] = [dict(r) for r in conn.execute(
        "SELECT * FROM ops_maintenance WHERE flight_id = ?", (flight_id,)
    ).fetchall()]

    result["retail"] = [dict(r) for r in conn.execute(
        "SELECT * FROM ops_retail WHERE flight_id = ?", (flight_id,)
    ).fetchall()]

    result["assigned_staff"] = [dict(r) for r in conn.execute(
        """SELECT DISTINCT s.staff_id, s.name, s.role, ge.event_type, ge.timestamp
           FROM ops_gate_events ge
           JOIN ops_staff s ON ge.staff_id = s.staff_id
           WHERE ge.flight_id = ?""",
        (flight_id,),
    ).fetchall()]

    conn.close()
    return result


def get_ops_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or (Path(__file__).parent / "flights.db")
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    tables = [
        "ops_flights", "ops_staff", "ops_shifts", "ops_gate_events",
        "ops_passengers", "ops_baggage", "ops_security", "ops_maintenance",
        "ops_retail",
    ]
    summary = {}
    for t in tables:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            summary[t] = count
        except Exception:
            summary[t] = 0
    conn.close()
    summary["loaded"] = any(v > 0 for v in summary.values())
    return summary
