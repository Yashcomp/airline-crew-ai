from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.models import Role

DEFAULT_DB = Path(__file__).parent.parent / "flights.db"


def get_staff_role_distribution(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT role, COUNT(*) as cnt FROM ops_staff GROUP BY role ORDER BY cnt DESC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM ops_staff").fetchone()[0]
        return {
            "loaded": True,
            "total_staff": total,
            "by_role": {r["role"]: r["cnt"] for r in rows},
            "role_fractions": {r["role"]: round(r["cnt"] / total, 3) for r in rows} if total > 0 else {},
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_shift_coverage_analysis(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    try:
        shift_rows = conn.execute("""
            SELECT
                CAST(substr(shift_start, 12, 2) AS INTEGER) as start_hour,
                role,
                COUNT(*) as staff_count
            FROM ops_shifts s
            JOIN ops_staff st ON s.staff_id = st.staff_id
            GROUP BY start_hour, role
            ORDER BY start_hour
        """).fetchall()

        hourly_by_role: Dict[int, Dict[str, int]] = {}
        for r in shift_rows:
            h = r["start_hour"]
            role = r["role"]
            hourly_by_role.setdefault(h, {})
            hourly_by_role[h][role] = hourly_by_role[h].get(role, 0) + r["staff_count"]

        flight_rows = conn.execute("""
            SELECT
                CAST(strftime('%H', std) AS INTEGER) as hour,
                COUNT(*) as flight_count,
                SUM(booked_pax) as total_pax
            FROM ops_flights
            GROUP BY hour ORDER BY hour
        """).fetchall()

        hourly_flights = {r["hour"]: {"flights": r["flight_count"], "pax": r["total_pax"]} for r in flight_rows}

        coverage_by_hour = []
        all_hours = sorted(set(list(hourly_by_role.keys()) + list(hourly_flights.keys())))
        for h in all_hours:
            staff = hourly_by_role.get(h, {})
            flights = hourly_flights.get(h, {})
            total_staff = sum(staff.values())
            flight_count = flights.get("flights", 0)
            pax = flights.get("pax", 0) or 0
            coverage_by_hour.append({
                "hour": h,
                "staff_on_duty": total_staff,
                "flights": flight_count,
                "pax": pax,
                "staff_per_flight": round(total_staff / flight_count, 1) if flight_count > 0 else 0,
                "staff_by_role": staff,
            })

        return {"loaded": True, "coverage_by_hour": coverage_by_hour}
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def identify_understaffed_periods(db_path: Optional[Path] = None, min_staff_per_flight: float = 2.0) -> List[Dict[str, Any]]:
    coverage = get_shift_coverage_analysis(db_path)
    if not coverage.get("loaded"):
        return []
    understaffed = []
    for hour_data in coverage.get("coverage_by_hour", []):
        if hour_data["flights"] > 0 and hour_data["staff_per_flight"] < min_staff_per_flight:
            understaffed.append({
                "hour": hour_data["hour"],
                "staff": hour_data["staff_on_duty"],
                "flights": hour_data["flights"],
                "staff_per_flight": hour_data["staff_per_flight"],
                "recommended": max(int(hour_data["flights"] * min_staff_per_flight), hour_data["staff_on_duty"] + 1),
            })
    return understaffed


def get_staff_utilization(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    try:
        staff_ids = [r["staff_id"] for r in conn.execute("SELECT staff_id FROM ops_staff").fetchall()]
        total_shifts = conn.execute("SELECT COUNT(*) FROM ops_shifts").fetchone()[0]
        total_gate = conn.execute("SELECT COUNT(DISTINCT staff_id) FROM ops_gate_events").fetchone()[0]
        total_security = conn.execute("SELECT COUNT(DISTINCT staff_id) FROM ops_security").fetchone()[0]
        total_maint = conn.execute("SELECT COUNT(DISTINCT staff_id) FROM ops_maintenance").fetchone()[0]
        total_retail = conn.execute("SELECT COUNT(DISTINCT staff_id) FROM ops_retail").fetchone()[0]

        assigned_to_ops = set()
        for r in conn.execute("SELECT staff_id FROM ops_gate_events").fetchall():
            assigned_to_ops.add(r["staff_id"])
        for r in conn.execute("SELECT staff_id FROM ops_security").fetchall():
            assigned_to_ops.add(r["staff_id"])
        for r in conn.execute("SELECT staff_id FROM ops_maintenance").fetchall():
            assigned_to_ops.add(r["staff_id"])
        for r in conn.execute("SELECT staff_id FROM ops_retail").fetchall():
            assigned_to_ops.add(r["staff_id"])

        idle = [s for s in staff_ids if s not in assigned_to_ops]

        return {
            "loaded": True,
            "total_staff": len(staff_ids),
            "assigned_to_gate_events": total_gate,
            "assigned_to_security": total_security,
            "assigned_to_maintenance": total_maint,
            "assigned_to_retail": total_retail,
            "idle_count": len(idle),
            "idle_staff": idle[:20],
            "total_shifts": total_shifts,
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_staff_shift_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM ops_shifts").fetchone()[0]
        avg_hours = conn.execute("SELECT AVG(shift_hours) FROM ops_shifts").fetchone()[0] or 0
        overtime = conn.execute("SELECT SUM(overtime) FROM ops_shifts").fetchone()[0] or 0
        by_date = conn.execute("""
            SELECT shift_date, COUNT(*) as shifts, AVG(shift_hours) as avg_hrs
            FROM ops_shifts GROUP BY shift_date ORDER BY shift_date
        """).fetchall()
        return {
            "loaded": True,
            "total_shifts": total,
            "avg_shift_hours": round(avg_hours, 1),
            "overtime_shifts": overtime,
            "daily_breakdown": [
                {"date": r["shift_date"], "shifts": r["shifts"], "avg_hours": round(r["avg_hrs"], 1)}
                for r in by_date
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()
