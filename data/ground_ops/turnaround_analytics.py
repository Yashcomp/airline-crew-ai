from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB = Path(__file__).parent.parent / "flights.db"


def get_turnaround_profile(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists() or not flight_id:
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        flight = conn.execute(
            "SELECT * FROM ops_flights WHERE flight_id = ?", (flight_id,)
        ).fetchone()
        if not flight:
            return {"loaded": False, "error": f"Flight {flight_id} not found"}

        result = dict(flight)

        result["gate_events"] = [dict(r) for r in conn.execute(
            "SELECT * FROM ops_gate_events WHERE flight_id = ?", (flight_id,)
        ).fetchall()]

        result["baggage"] = [dict(r) for r in conn.execute(
            "SELECT * FROM ops_baggage WHERE flight_id = ?", (flight_id,)
        ).fetchall()]

        result["maintenance"] = [dict(r) for r in conn.execute(
            "SELECT * FROM ops_maintenance WHERE flight_id = ?", (flight_id,)
        ).fetchall()]

        bags = result["baggage"]
        total_weight = sum(b.get("weight", 0) or 0 for b in bags)
        total_pieces = sum(b.get("pieces", 0) or 0 for b in bags)
        delayed_bags = sum(1 for b in bags if b.get("delay"))

        result["baggage_summary"] = {
            "total_bags": len(bags),
            "total_weight_kg": round(total_weight, 1),
            "total_pieces": total_pieces,
            "delayed_bags": delayed_bags,
        }

        gate_evts = result["gate_events"]
        if gate_evts:
            result["boarding_duration_min"] = gate_evts[0].get("duration_min", 0)
            result["assigned_ground_staff"] = gate_evts[0].get("staff_id", "")
        else:
            result["boarding_duration_min"] = 0
            result["assigned_ground_staff"] = ""

        maint = result["maintenance"]
        result["maintenance_issues"] = len(maint)
        result["airworthy"] = all(m.get("airworthy") for m in maint) if maint else True

        return result
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_boarding_efficiency(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT ge.flight_id, ge.duration_min,
                   COUNT(DISTINCT p.passenger_id) as pax_count,
                   f.booked_pax, f.aircraft_type
            FROM ops_gate_events ge
            JOIN ops_flights f ON ge.flight_id = f.flight_id
            LEFT JOIN ops_passengers p ON ge.flight_id = p.flight_id
            WHERE ge.event_type = 'Boarding Start'
            GROUP BY ge.flight_id, ge.duration_min, f.booked_pax, f.aircraft_type
        """).fetchall()

        if not rows:
            return {"loaded": True, "flights_analyzed": 0, "avg_boarding_min": 0}

        flights = [dict(r) for r in rows]
        avg_duration = sum(f["duration_min"] for f in flights) / len(flights)
        avg_pax = sum(f["pax_count"] for f in flights) / len(flights)

        return {
            "loaded": True,
            "flights_analyzed": len(flights),
            "avg_boarding_min": round(avg_duration, 1),
            "avg_pax_per_flight": round(avg_pax, 1),
            "efficiency_per_pax": round(avg_duration / avg_pax, 2) if avg_pax > 0 else 0,
            "flights": flights[:10],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_maintenance_impact(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        ac_rows = conn.execute("""
            SELECT aircraft_reg,
                   COUNT(*) as work_orders,
                   AVG(duration_hrs) as avg_duration,
                   SUM(CASE WHEN airworthy = 0 THEN 1 ELSE 0 END) as not_airworthy,
                   AVG(severity) as avg_severity
            FROM ops_maintenance
            GROUP BY aircraft_reg
            ORDER BY work_orders DESC
        """).fetchall()

        defect_rows = conn.execute("""
            SELECT defect, component, COUNT(*) as cnt, AVG(duration_hrs) as avg_hrs
            FROM ops_maintenance
            GROUP BY defect, component
            ORDER BY cnt DESC
        """).fetchall()

        return {
            "loaded": True,
            "aircraft_maintenance": [dict(r) for r in ac_rows],
            "top_defects": [dict(r) for r in defect_rows[:10]],
            "total_work_orders": sum(r["work_orders"] for r in ac_rows),
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()
