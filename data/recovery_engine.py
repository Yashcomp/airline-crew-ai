from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path(__file__).parent / "flights.db"


def assess_disruption_impact(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
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

        passengers = conn.execute(
            "SELECT * FROM ops_passengers WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        pax_list = [dict(r) for r in passengers]
        connecting_pax = [p for p in pax_list if p.get("is_connecting")]

        baggage = conn.execute(
            "SELECT * FROM ops_baggage WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        bag_list = [dict(r) for r in baggage]
        loaded_bags = [b for b in bag_list if b.get("status") == "Loaded"]

        gate_events = conn.execute(
            "SELECT * FROM ops_gate_events WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        staff_assigned = [dict(r).get("staff_id", "") for r in gate_events]

        maintenance = conn.execute(
            "SELECT * FROM ops_maintenance WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        maint_list = [dict(r) for r in maintenance]

        retail = conn.execute(
            "SELECT SUM(amount) as total FROM ops_retail WHERE flight_id = ?", (flight_id,)
        ).fetchone()

        security = conn.execute("""
            SELECT COUNT(*) as cleared
            FROM ops_security s
            JOIN ops_passengers p ON s.passenger_id = p.passenger_id
            WHERE p.flight_id = ? AND s.result = 'Clear'
        """, (flight_id,)).fetchone()

        disruption_reason = dict(flight).get("delay_reason", "Unknown")
        delay_min = dict(flight).get("delay_min", 0)

        impact = {
            "loaded": True,
            "flight_id": flight_id,
            "origin": dict(flight)["origin"],
            "destination": dict(flight)["destination"],
            "aircraft": dict(flight)["aircraft_type"],
            "delay_min": delay_min,
            "disruption_reason": disruption_reason,
            "passengers_affected": len(pax_list),
            "connecting_passengers_at_risk": len(connecting_pax),
            "baggage_total": len(bag_list),
            "baggage_loaded": len(loaded_bags),
            "baggage_needs_reroute": len(loaded_bags),
            "ground_staff_assigned": len(staff_assigned),
            "staff_ids": staff_assigned,
            "maintenance_issues": len(maint_list),
            "airworthy": all(m.get("airworthy") for m in maint_list) if maint_list else True,
            "retail_revenue_impact": round(retail["total"] or 0, 0),
            "passengers_cleared_security": security["cleared"] if security else 0,
        }

        if connecting_pax:
            connecting_routes = {}
            for p in connecting_pax:
                dest = p.get("gate", "Unknown")
                connecting_routes[dest] = connecting_routes.get(dest, 0) + 1
            impact["connecting_flow"] = connecting_routes

        return impact
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"loaded": False, "error": str(e)}
    finally:
        conn.close()


def find_recovery_options(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
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

        flight_dict = dict(flight)

        staff_recs = conn.execute("""
            SELECT st.staff_id, st.name, st.role, s.shift_start, s.shift_end
            FROM ops_staff st
            JOIN ops_shifts s ON st.staff_id = s.staff_id
            WHERE s.shift_date = date(?)
            AND st.staff_id NOT IN (
                SELECT staff_id FROM ops_gate_events WHERE flight_id = ?
            )
            ORDER BY s.shift_start
        """, (flight_dict["std"][:10] if flight_dict.get("std") else "", flight_id)).fetchall()

        available_staff = [dict(r) for r in staff_recs]

        gate_events = conn.execute(
            "SELECT * FROM ops_gate_events WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        assigned_staff = [dict(r).get("staff_id", "") for r in gate_events]

        baggage = conn.execute(
            "SELECT * FROM ops_baggage WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        loaded_bags = [dict(r) for r in baggage if dict(r).get("status") == "Loaded"]

        maintenance = conn.execute(
            "SELECT * FROM ops_maintenance WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        maint_issues = [dict(r) for r in maintenance]

        recommendations = []

        if available_staff:
            recommendations.append({
                "type": "staff",
                "message": f"{len(available_staff)} staff available for this flight",
                "candidates": available_staff[:5],
            })

        if loaded_bags:
            recommendations.append({
                "type": "baggage",
                "message": f"{len(loaded_bags)} bags need rerouting to new flight",
                "bags": [b["baggage_tag"] for b in loaded_bags[:10]],
            })

        if maint_issues:
            airworthy_issues = [m for m in maint_issues if not m.get("airworthy")]
            if airworthy_issues:
                recommendations.append({
                    "type": "maintenance",
                    "message": f"Aircraft has {len(airworthy_issues)} airworthiness issues — consider AOG",
                    "issues": airworthy_issues,
                })

        if not recommendations:
            recommendations.append({
                "type": "info",
                "message": "No immediate recovery actions identified from available data.",
            })

        return {
            "loaded": True,
            "flight_id": flight_id,
            "available_staff_count": len(available_staff),
            "loaded_bags_to_reroute": len(loaded_bags),
            "maintenance_issues": len(maint_issues),
            "recommendations": recommendations,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"loaded": False, "error": str(e)}
    finally:
        conn.close()


def get_disruption_cascade(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
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

        flight_dict = dict(flight)
        aircraft_reg = None
        maint = conn.execute(
            "SELECT aircraft_reg FROM ops_maintenance WHERE flight_id = ?", (flight_id,)
        ).fetchone()
        if maint:
            aircraft_reg = dict(maint)["aircraft_reg"]

        affected_flights = []
        if aircraft_reg:
            other_flights = conn.execute(
                "SELECT flight_id, origin, destination, std FROM ops_flights WHERE flight_id != ? AND registration = ?",
                (flight_id, aircraft_reg)
            ).fetchall()
            affected_flights = [dict(r) for r in other_flights]

        affected_staff = conn.execute(
            "SELECT staff_id FROM ops_gate_events WHERE flight_id = ?", (flight_id,)
        ).fetchall()
        staff_ids = [dict(r)["staff_id"] for r in affected_staff]

        staff_other_flights = []
        for sid in staff_ids:
            others = conn.execute("""
                SELECT DISTINCT ge.flight_id
                FROM ops_gate_events ge
                WHERE ge.staff_id = ? AND ge.flight_id != ?
            """, (sid, flight_id)).fetchall()
            for r in others:
                staff_other_flights.append({"staff_id": sid, "flight_id": dict(r)["flight_id"]})

        affected_pax = conn.execute(
            "SELECT COUNT(*) FROM ops_passengers WHERE flight_id = ?", (flight_id,)
        ).fetchone()[0]

        return {
            "loaded": True,
            "flight_id": flight_id,
            "aircraft_reg": aircraft_reg,
            "affected_flights_same_aircraft": affected_flights,
            "affected_staff": staff_ids,
            "staff_other_flights": staff_other_flights,
            "passengers_affected": affected_pax,
            "cascade_severity": "High" if len(affected_flights) > 3 else "Medium" if len(affected_flights) > 0 else "Low",
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"loaded": False, "error": str(e)}
    finally:
        conn.close()
