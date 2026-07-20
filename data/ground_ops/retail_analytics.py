from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB = Path(__file__).parent.parent / "flights.db"


def get_revenue_by_flight(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT flight_id, COUNT(*) as transactions,
                   SUM(amount) as total_revenue, AVG(amount) as avg_amount
            FROM ops_retail
            GROUP BY flight_id ORDER BY total_revenue DESC
        """).fetchall()

        total_rev = conn.execute("SELECT SUM(amount) FROM ops_retail").fetchone()[0] or 0
        total_txns = conn.execute("SELECT COUNT(*) FROM ops_retail").fetchone()[0] or 0

        return {
            "loaded": True,
            "total_revenue": round(total_rev, 0),
            "total_transactions": total_txns,
            "avg_revenue_per_txn": round(total_rev / total_txns, 0) if total_txns > 0 else 0,
            "top_flights": [
                {"flight_id": r["flight_id"], "transactions": r["transactions"],
                 "total_revenue": round(r["total_revenue"], 0),
                 "avg_amount": round(r["avg_amount"], 0)}
                for r in rows[:20]
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_revenue_by_gate(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT terminal, location, COUNT(*) as transactions,
                   SUM(amount) as total_revenue
            FROM ops_retail
            GROUP BY terminal, location ORDER BY total_revenue DESC
        """).fetchall()

        return {
            "loaded": True,
            "by_location": [
                {"terminal": r["terminal"], "location": r["location"],
                 "transactions": r["transactions"],
                 "total_revenue": round(r["total_revenue"], 0)}
                for r in rows
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_passenger_spend_profile(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        class_rows = conn.execute("""
            SELECT p.seat_class, COUNT(r.txn_id) as transactions,
                   SUM(r.amount) as total_spend, AVG(r.amount) as avg_spend
            FROM ops_retail r
            JOIN ops_passengers p ON r.passenger_id = p.passenger_id
            GROUP BY p.seat_class
        """).fetchall()

        age_rows = conn.execute("""
            SELECT p.age_category, COUNT(r.txn_id) as transactions,
                   SUM(r.amount) as total_spend, AVG(r.amount) as avg_spend
            FROM ops_retail r
            JOIN ops_passengers p ON r.passenger_id = p.passenger_id
            GROUP BY p.age_category
        """).fetchall()

        loyalty_rows = conn.execute("""
            SELECT
                CASE
                    WHEN p.loyalty_score < 1 THEN 'Low (<1)'
                    WHEN p.loyalty_score < 3 THEN 'Medium (1-3)'
                    ELSE 'High (3+)'
                END as tier,
                COUNT(r.txn_id) as transactions,
                AVG(r.amount) as avg_spend
            FROM ops_retail r
            JOIN ops_passengers p ON r.passenger_id = p.passenger_id
            GROUP BY tier
        """).fetchall()

        return {
            "loaded": True,
            "by_class": [
                {"class": r["seat_class"], "transactions": r["transactions"],
                 "total_spend": round(r["total_spend"], 0),
                 "avg_spend": round(r["avg_spend"], 0)}
                for r in class_rows
            ],
            "by_age": [
                {"age": r["age_category"], "transactions": r["transactions"],
                 "total_spend": round(r["total_spend"], 0),
                 "avg_spend": round(r["avg_spend"], 0)}
                for r in age_rows
            ],
            "by_loyalty": [
                {"tier": r["tier"], "transactions": r["transactions"],
                 "avg_spend": round(r["avg_spend"], 0)}
                for r in loyalty_rows
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def predict_retail_demand(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    try:
        avg_txn_per_pax = conn.execute("""
            SELECT COUNT(DISTINCT r.txn_id) * 1.0 / COUNT(DISTINCT p.passenger_id)
            FROM ops_retail r
            JOIN ops_passengers p ON r.passenger_id = p.passenger_id
        """).fetchone()[0] or 0.5

        avg_spend = conn.execute("SELECT AVG(amount) FROM ops_retail").fetchone()[0] or 2000

        if flight_id:
            pax = conn.execute(
                "SELECT COUNT(*) FROM ops_passengers WHERE flight_id = ?", (flight_id,)
            ).fetchone()[0]
        else:
            pax = conn.execute("SELECT COUNT(*) FROM ops_passengers").fetchone()[0]

        predicted_txns = int(pax * avg_txn_per_pax)
        predicted_revenue = round(predicted_txns * avg_spend, 0)

        return {
            "loaded": True,
            "flight_id": flight_id,
            "passengers": pax,
            "predicted_transactions": predicted_txns,
            "predicted_revenue": predicted_revenue,
            "avg_txn_per_pax": round(avg_txn_per_pax, 2),
            "avg_spend_per_txn": round(avg_spend, 0),
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()
