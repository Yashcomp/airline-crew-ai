from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB = Path(__file__).parent.parent / "flights.db"


def get_security_throughput(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT COUNT(*) FROM ops_security").fetchone()[0]
        if total == 0:
            return {"loaded": True, "total_screenings": 0}

        avg_time = conn.execute("SELECT AVG(processing_time) FROM ops_security").fetchone()[0] or 0
        avg_pass_per_hour = conn.execute("SELECT AVG(pass_per_hour) FROM ops_security").fetchone()[0] or 0
        avg_utilization = conn.execute("SELECT AVG(utilization) FROM ops_security").fetchone()[0] or 0

        type_rows = conn.execute("""
            SELECT screen_type, COUNT(*) as cnt, AVG(processing_time) as avg_time
            FROM ops_security GROUP BY screen_type ORDER BY cnt DESC
        """).fetchall()

        staff_rows = conn.execute("""
            SELECT staff_id, COUNT(*) as screenings, AVG(processing_time) as avg_time
            FROM ops_security GROUP BY staff_id ORDER BY screenings DESC LIMIT 10
        """).fetchall()

        return {
            "loaded": True,
            "total_screenings": total,
            "avg_processing_time": round(avg_time, 1),
            "avg_pass_per_hour": round(avg_pass_per_hour, 0),
            "avg_utilization": round(avg_utilization, 2),
            "by_screen_type": [
                {"type": r["screen_type"], "count": r["cnt"], "avg_time": round(r["avg_time"], 1)}
                for r in type_rows
            ],
            "top_staff": [
                {"staff_id": r["staff_id"], "screenings": r["screenings"], "avg_time": round(r["avg_time"], 1)}
                for r in staff_rows
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_screening_staff_performance(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT s.staff_id, st.name, st.role,
                   COUNT(*) as screenings,
                   AVG(s.processing_time) as avg_time,
                   SUM(CASE WHEN s.has_alarm THEN 1 ELSE 0 END) as alarms
            FROM ops_security s
            JOIN ops_staff st ON s.staff_id = st.staff_id
            GROUP BY s.staff_id
            ORDER BY screenings DESC
        """).fetchall()

        return {
            "loaded": True,
            "staff": [
                {
                    "staff_id": r["staff_id"],
                    "name": r["name"],
                    "role": r["role"],
                    "screenings": r["screenings"],
                    "avg_time": round(r["avg_time"], 1),
                    "alarms": r["alarms"],
                }
                for r in rows
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def predict_queue_buildup(db_path: Optional[Path] = None, departure_window_hours: int = 4) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        flight_rows = conn.execute("""
            SELECT CAST(strftime('%H', std) AS INTEGER) as hour,
                   COUNT(*) as flights,
                   SUM(booked_pax) as pax
            FROM ops_flights
            GROUP BY hour ORDER BY hour
        """).fetchall()

        hourly = {r["hour"]: {"flights": r["flights"], "pax": r["pax"] or 0} for r in flight_rows}

        avg_processing = conn.execute(
            "SELECT AVG(processing_time) FROM ops_security"
        ).fetchone()[0] or 60

        avg_capacity = conn.execute(
            "SELECT AVG(capacity) FROM ops_security"
        ).fetchone()[0] or 200

        predictions = []
        for h in range(24):
            data = hourly.get(h, {"flights": 0, "pax": 0})
            pax = data["pax"]
            throughput_per_hour = avg_capacity
            queue_buildup = max(0, pax - throughput_per_hour)
            predicted_wait_min = round(queue_buildup * avg_processing / throughput_per_hour, 1) if throughput_per_hour > 0 else 0

            predictions.append({
                "hour": h,
                "flights": data["flights"],
                "passengers": pax,
                "predicted_queue_length": int(queue_buildup),
                "predicted_wait_min": predicted_wait_min,
                "risk_level": "High" if predicted_wait_min > 15 else "Medium" if predicted_wait_min > 5 else "Low",
            })

        return {"loaded": True, "predictions": predictions}
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()
