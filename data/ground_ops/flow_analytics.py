from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = Path(__file__).parent.parent / "flights.db"


def get_passenger_profile(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if flight_id:
            rows = conn.execute(
                "SELECT * FROM ops_passengers WHERE flight_id = ?", (flight_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM ops_passengers LIMIT 500").fetchall()

        passengers = [dict(r) for r in rows]

        nationality_dist: Dict[str, int] = {}
        age_dist: Dict[str, int] = {}
        class_dist: Dict[str, int] = {}
        total_loyalty = 0.0
        connecting_count = 0

        for p in passengers:
            nat = p.get("nationality", "Unknown")
            nationality_dist[nat] = nationality_dist.get(nat, 0) + 1
            age = p.get("age_category", "Unknown")
            age_dist[age] = age_dist.get(age, 0) + 1
            cls = p.get("seat_class", "Unknown")
            class_dist[cls] = class_dist.get(cls, 0) + 1
            total_loyalty += p.get("loyalty_score", 0) or 0
            if p.get("is_connecting"):
                connecting_count += 1

        total = len(passengers)
        return {
            "loaded": True,
            "flight_id": flight_id,
            "total_passengers": total,
            "nationality_distribution": nationality_dist,
            "age_distribution": age_dist,
            "class_distribution": class_dist,
            "avg_loyalty_score": round(total_loyalty / total, 2) if total > 0 else 0,
            "connecting_passengers": connecting_count,
            "connecting_rate": round(connecting_count / total, 3) if total > 0 else 0,
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_baggage_load_profile(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        if flight_id:
            rows = conn.execute(
                "SELECT * FROM ops_baggage WHERE flight_id = ?", (flight_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM ops_baggage LIMIT 1000").fetchall()

        bags = [dict(r) for r in rows]
        total_bags = len(bags)
        total_weight = sum(b.get("weight", 0) or 0 for b in bags)
        total_pieces = sum(b.get("pieces", 0) or 0 for b in bags)
        delayed_bags = sum(1 for b in bags if b.get("delay"))
        oversized = sum(1 for b in bags if b.get("is_oversized"))

        status_dist: Dict[str, int] = {}
        for b in bags:
            st = b.get("status", "Unknown")
            status_dist[st] = status_dist.get(st, 0) + 1

        return {
            "loaded": True,
            "flight_id": flight_id,
            "total_bags": total_bags,
            "total_weight_kg": round(total_weight, 1),
            "total_pieces": total_pieces,
            "avg_weight_per_bag": round(total_weight / total_bags, 1) if total_bags > 0 else 0,
            "delayed_bags": delayed_bags,
            "oversized_bags": oversized,
            "status_distribution": status_dist,
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_demand_by_route(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        route_rows = conn.execute("""
            SELECT f.flight_id, f.origin, f.destination, f.route_type,
                   COUNT(p.passenger_id) as pax_count,
                   AVG(p.loyalty_score) as avg_loyalty,
                   SUM(CASE WHEN p.is_connecting THEN 1 ELSE 0 END) as connecting
            FROM ops_flights f
            LEFT JOIN ops_passengers p ON f.flight_id = p.flight_id
            GROUP BY f.flight_id, f.origin, f.destination, f.route_type
            HAVING pax_count > 0
        """).fetchall()

        routes: Dict[str, Dict[str, Any]] = {}
        for r in route_rows:
            key = f"{r['origin']}-{r['destination']}"
            if key not in routes:
                routes[key] = {
                    "flights": 0, "total_pax": 0, "total_connecting": 0,
                    "route_type": r["route_type"], "loyalty_sum": 0.0,
                }
            routes[key]["flights"] += 1
            routes[key]["total_pax"] += r["pax_count"]
            routes[key]["total_connecting"] += r["connecting"] or 0
            routes[key]["loyalty_sum"] += r["avg_loyalty"] or 0

        for k, v in routes.items():
            v["avg_pax_per_flight"] = round(v["total_pax"] / v["flights"], 1) if v["flights"] > 0 else 0
            v["avg_loyalty"] = round(v["loyalty_sum"] / v["flights"], 2) if v["flights"] > 0 else 0
            v["connecting_rate"] = round(v["total_connecting"] / v["total_pax"], 3) if v["total_pax"] > 0 else 0
            del v["loyalty_sum"]

        return {"loaded": True, "routes": routes}
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def predict_baggage_load(db_path: Optional[Path] = None, flight_id: Optional[str] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    try:
        if flight_id:
            pax = conn.execute(
                "SELECT COUNT(*) FROM ops_passengers WHERE flight_id = ?", (flight_id,)
            ).fetchone()[0]
            avg_bags = conn.execute(
                "SELECT AVG(baggage_count) FROM ops_passengers WHERE flight_id = ?", (flight_id,)
            ).fetchone()[0] or 1.5
        else:
            pax = conn.execute("SELECT COUNT(*) FROM ops_passengers").fetchone()[0]
            avg_bags = conn.execute("SELECT AVG(baggage_count) FROM ops_passengers").fetchone()[0] or 1.5

        avg_weight = conn.execute("SELECT AVG(weight) FROM ops_baggage").fetchone()[0] or 15.0
        avg_pieces = conn.execute("SELECT AVG(pieces) FROM ops_baggage").fetchone()[0] or 2.0

        predicted_bags = int(pax * avg_bags)
        predicted_weight = round(predicted_bags * avg_weight, 1)
        predicted_pieces = round(predicted_bags * avg_pieces, 0)

        return {
            "loaded": True,
            "flight_id": flight_id,
            "passengers": pax,
            "predicted_bags": predicted_bags,
            "predicted_weight_kg": predicted_weight,
            "predicted_pieces": int(predicted_pieces),
            "avg_bags_per_pax": round(avg_bags, 2),
            "avg_weight_per_bag": round(avg_weight, 1),
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()


def get_connecting_pax_analysis(db_path: Optional[Path] = None) -> Dict[str, Any]:
    path = db_path or DEFAULT_DB
    if not path.exists():
        return {"loaded": False}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        total = conn.execute("SELECT COUNT(*) FROM ops_passengers").fetchone()[0]
        connecting = conn.execute(
            "SELECT COUNT(*) FROM ops_passengers WHERE is_connecting = 1"
        ).fetchone()[0]

        by_route = conn.execute("""
            SELECT f.origin || '-' || f.destination as route,
                   COUNT(*) as total,
                   SUM(CASE WHEN p.is_connecting THEN 1 ELSE 0 END) as connecting
            FROM ops_passengers p
            JOIN ops_flights f ON p.flight_id = f.flight_id
            GROUP BY route
            HAVING total > 5
            ORDER BY connecting DESC
            LIMIT 10
        """).fetchall()

        return {
            "loaded": True,
            "total_passengers": total,
            "connecting_passengers": connecting,
            "connecting_rate": round(connecting / total, 3) if total > 0 else 0,
            "top_connecting_routes": [
                {"route": r["route"], "total": r["total"], "connecting": r["connecting"],
                 "rate": round(r["connecting"] / r["total"], 3) if r["total"] > 0 else 0}
                for r in by_route
            ],
        }
    except Exception:
        return {"loaded": False}
    finally:
        conn.close()
