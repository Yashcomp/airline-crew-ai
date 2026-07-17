from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


_route_demand_base = {
    ("DEL", "BOM"): 280, ("BOM", "DEL"): 275,
    ("DEL", "CCU"): 200, ("CCU", "DEL"): 195,
    ("DEL", "BLR"): 220, ("BLR", "DEL"): 215,
    ("DEL", "MAA"): 180, ("MAA", "DEL"): 175,
    ("DEL", "HYD"): 190, ("HYD", "DEL"): 185,
    ("BOM", "BLR"): 160, ("BLR", "BOM"): 155,
    ("BOM", "CCU"): 140, ("CCU", "BOM"): 135,
    ("BOM", "MAA"): 130, ("MAA", "BOM"): 125,
    ("BLR", "CCU"): 110, ("CCU", "BLR"): 105,
    ("BLR", "HYD"): 120, ("HYD", "BLR"): 115,
}

_hourly_profile = {
    0: 0.02, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.03, 5: 0.06,
    6: 0.08, 7: 0.09, 8: 0.10, 9: 0.09, 10: 0.08, 11: 0.07,
    12: 0.06, 13: 0.05, 14: 0.04, 15: 0.04, 16: 0.05, 17: 0.06,
    18: 0.07, 19: 0.08, 20: 0.07, 21: 0.05, 22: 0.04, 23: 0.03,
}

_day_of_week_multiplier = {
    0: 1.15, 1: 1.10, 2: 1.05, 3: 1.00, 4: 1.20, 5: 1.35, 6: 1.25,
}


def forecast_demand(
    origin: str,
    destination: str,
    target_date: Optional[datetime] = None,
    hours_ahead: int = 24,
) -> Dict[str, Any]:
    if target_date is None:
        target_date = datetime.now()

    route_key = (origin.upper(), destination.upper())
    base_pax = _route_demand_base.get(route_key, 120)

    dow = target_date.weekday()
    dow_mult = _day_of_week_multiplier.get(dow, 1.0)

    hourly_demand = []
    for h in range(24):
        hour_mult = _hourly_profile.get(h, 0.04)
        noise = random.uniform(0.9, 1.1)
        pax = base_pax * dow_mult * hour_mult * 4 * noise
        hourly_demand.append({
            "hour": h,
            "estimated_pax": round(pax),
            "load_factor": round(min(pax / 180, 1.0), 2),
        })

    peak_hours = sorted(hourly_demand, key=lambda x: x["estimated_pax"], reverse=True)[:5]
    total_daily = sum(h["estimated_pax"] for h in hourly_demand)
    avg_load = round(total_daily / (24 * 180), 2)

    flights_needed = max(1, math.ceil(total_daily / 160))

    return {
        "route": f"{origin.upper()} -> {destination.upper()}",
        "date": target_date.strftime("%Y-%m-%d"),
        "day_of_week": target_date.strftime("%A"),
        "base_demand": base_pax,
        "day_multiplier": dow_mult,
        "total_daily_pax": round(total_daily),
        "avg_load_factor": avg_load,
        "flights_needed": flights_needed,
        "peak_hours": peak_hours,
        "hourly_breakdown": hourly_demand,
    }


def forecast_multi_route(
    routes: List[Tuple[str, str]],
    target_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    forecasts = {}
    total_pax = 0
    total_flights = 0

    for origin, dest in routes:
        f = forecast_demand(origin, dest, target_date)
        key = f"{origin}-{dest}"
        forecasts[key] = {
            "total_pax": f["total_daily_pax"],
            "flights_needed": f["flights_needed"],
            "avg_load": f["avg_load_factor"],
        }
        total_pax += f["total_daily_pax"]
        total_flights += f["flights_needed"]

    return {
        "date": (target_date or datetime.now()).strftime("%Y-%m-%d"),
        "route_forecasts": forecasts,
        "summary": {
            "total_routes": len(routes),
            "total_estimated_pax": total_pax,
            "total_flights_needed": total_flights,
        },
    }


def get_demand_summary(db_path: Optional[Path] = None) -> Dict[str, Any]:
    from data.flights_db import get_flights
    flights = get_flights(db_path=db_path)
    if not flights:
        return {"message": "No flight data available."}

    total_pax = sum(f.pax_count for f in flights)
    by_route: Dict[str, Dict[str, Any]] = {}
    by_hour: Dict[int, int] = {}

    for f in flights:
        route = f"{f.origin}-{f.destination}"
        by_route.setdefault(route, {"flights": 0, "total_pax": 0, "avg_pax": 0})
        by_route[route]["flights"] += 1
        by_route[route]["total_pax"] += f.pax_count

        by_hour.setdefault(f.std.hour, 0)
        by_hour[f.std.hour] += f.pax_count

    for route_data in by_route.values():
        if route_data["flights"] > 0:
            route_data["avg_pax"] = round(route_data["total_pax"] / route_data["flights"])

    peak_hours = sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_flights": len(flights),
        "total_pax": total_pax,
        "avg_pax_per_flight": round(total_pax / len(flights)) if flights else 0,
        "route_breakdown": by_route,
        "peak_demand_hours": [{"hour": h, "pax": p} for h, p in peak_hours],
    }
