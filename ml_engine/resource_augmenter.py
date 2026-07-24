from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from data.crew_loader import load_crew
from data.models import CrewMember, Role


def _utilization_score(member: CrewMember) -> float:
    duty_ratio = member.current_duty_hours / 12.0
    weekly_ratio = member.rolling_7_day_hours / 35.0
    night_factor = member.consecutive_night_shifts / 3.0
    rest_penalty = 1.0 if member.rest_status.lower() == "legal" else 0.0

    raw = (duty_ratio * 0.3 + weekly_ratio * 0.3 + night_factor * 0.2) * rest_penalty
    return round(min(raw, 1.0), 3)


def _fatigue_score(member: CrewMember) -> float:
    duty_load = member.current_duty_hours / 12.0
    weekly_load = member.rolling_7_day_hours / 35.0
    night_stress = member.consecutive_night_shifts / 3.0
    rest_gap = member.days_since_rest / 3.0
    consecutive = member.consecutive_days_on / 6.0

    raw = (duty_load * 0.25 + weekly_load * 0.25 + night_stress * 0.15
           + rest_gap * 0.2 + consecutive * 0.15)
    return round(min(raw, 1.0), 3)


def _availability_score(member: CrewMember) -> float:
    if member.rest_status.lower() != "legal":
        return 0.0
    if member.current_duty_hours >= 11.5:
        return 0.0
    if member.rolling_7_day_hours >= 34:
        return 0.1
    if member.consecutive_night_shifts >= 2:
        return 0.2
    return 1.0


def _cost_efficiency(member: CrewMember) -> float:
    effective_cost = member.base_cost * member.overtime_multiplier
    return round(1.0 / max(effective_cost, 1.0), 4)


def score_crew_utilization(csv_path: str | Path) -> List[Dict[str, Any]]:
    crew = load_crew(csv_path)
    scored = []
    for m in crew:
        util = _utilization_score(m)
        fatigue = _fatigue_score(m)
        avail = _availability_score(m)
        cost_eff = _cost_efficiency(m)

        composite = round(
            avail * 0.35 + (1 - fatigue) * 0.25 + util * 0.20 + cost_eff * 0.20, 3
        )

        status = "Available"
        if avail == 0.0:
            status = "Unavailable"
        elif avail < 0.5:
            status = "Limited"

        fatigue_risk = "Low"
        if fatigue > 0.7:
            fatigue_risk = "High"
        elif fatigue > 0.4:
            fatigue_risk = "Medium"

        scored.append({
            "crew_id": m.crew_id,
            "name": m.name,
            "role": m.role.value,
            "base_airport": m.base_airport,
            "utilization": util,
            "fatigue": fatigue,
            "fatigue_risk": fatigue_risk,
            "availability": avail,
            "cost_efficiency": cost_eff,
            "composite_score": composite,
            "status": status,
            "current_duty": m.current_duty_hours,
            "weekly_hours": m.rolling_7_day_hours,
            "night_shifts": m.consecutive_night_shifts,
            "rest_status": m.rest_status,
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def find_optimal_swaps(
    csv_path: str | Path,
    origin: str = "",
    aircraft_type: str = "",
    max_results: int = 5,
) -> List[Dict[str, Any]]:
    crew = load_crew(csv_path)
    scored = score_crew_utilization(csv_path)

    available = [s for s in scored if s["status"] == "Available"]
    unavailable = [s for s in scored if s["status"] != "Available" and s["fatigue"] > 0.5]

    swaps = []
    for tired in unavailable:
        for fresh in available:
            if fresh["role"] != tired["role"]:
                continue
            if origin and fresh["base_airport"] != origin.upper():
                continue

            cost_diff = 0
            for m in crew:
                if m.crew_id == tired["crew_id"]:
                    cost_diff -= m.base_cost * m.overtime_multiplier
                if m.crew_id == fresh["crew_id"]:
                    cost_diff += m.base_cost * m.overtime_multiplier

            fatigue_gain = round(tired["fatigue"] - fresh["fatigue"], 3)
            if fatigue_gain > 0.1:
                swaps.append({
                    "replace": tired["name"],
                    "replace_id": tired["crew_id"],
                    "replace_fatigue": tired["fatigue"],
                    "with": fresh["name"],
                    "with_id": fresh["crew_id"],
                    "with_fatigue": fresh["fatigue"],
                    "fatigue_improvement": fatigue_gain,
                    "cost_impact": round(cost_diff, 2),
                    "role": tired["role"],
                })

    swaps.sort(key=lambda x: x["fatigue_improvement"], reverse=True)
    return swaps[:max_results]


def forecast_crew_needs(
    csv_path: str | Path,
    flights_today: int = 10,
    avg_flight_hours: float = 2.0,
    disruption_rate: float = 0.15,
    today_schedule: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    crew = load_crew(csv_path)
    if not crew:
        return {"error": "No crew data available."}

    role_counts: Dict[str, int] = {}
    role_available: Dict[str, int] = {}
    for m in crew:
        role_counts[m.role.value] = role_counts.get(m.role.value, 0) + 1
        if _availability_score(m) > 0:
            role_available[m.role.value] = role_available.get(m.role.value, 0) + 1

    if today_schedule and len(today_schedule) > 0:
        ml_disruption_rate = sum(
            f.get("prediction", {}).get("delay_probability", 0)
            for f in today_schedule
        ) / len(today_schedule)
        disruption_rate = max(ml_disruption_rate, 0.05)
        flights_today = len(today_schedule)
        durations = [f.get("avg_duration_min", 120) for f in today_schedule]
        avg_flight_hours = sum(durations) / len(durations) / 60.0 if durations else avg_flight_hours

    total_flight_hours = flights_today * avg_flight_hours
    expected_disruptions = int(flights_today * disruption_rate)
    extra_crew_needed = expected_disruptions * 2

    captains_per_flight = 1
    fo_per_flight = 1
    cabin_per_flight = 2
    ground_per_flight = 1

    required = {
        "Captain": flights_today * captains_per_flight + extra_crew_needed,
        "FO": flights_today * fo_per_flight + extra_crew_needed,
        "CabinCrew": flights_today * cabin_per_flight + extra_crew_needed * 2,
        "GroundStaff": flights_today * ground_per_flight + extra_crew_needed,
    }

    gaps = {}
    for role, needed in required.items():
        available = role_available.get(role, 0)
        gap = needed - available
        gaps[role] = {
            "required": needed,
            "available": available,
            "total_pool": role_counts.get(role, 0),
            "gap": gap,
            "status": "Sufficient" if gap <= 0 else "Shortage",
        }

    total_required = sum(required.values())
    total_available = sum(role_available.values())
    utilization_pct = round((total_flight_hours / max(total_available * 8, 1)) * 100, 1)

    return {
        "flights_today": flights_today,
        "avg_flight_hours": avg_flight_hours,
        "total_flight_hours": total_flight_hours,
        "expected_disruptions": expected_disruptions,
        "role_breakdown": gaps,
        "overall": {
            "total_required": total_required,
            "total_available": total_available,
            "utilization_pct": utilization_pct,
            "status": "OK" if total_available >= total_required else "Understaffed",
        },
    }


def get_augmentation_report(csv_path: str | Path, db_path: Optional[Path] = None) -> Dict[str, Any]:
    scores = score_crew_utilization(csv_path)
    swaps = find_optimal_swaps(csv_path)
    forecast = forecast_crew_needs(csv_path)

    high_fatigue = [s for s in scores if s["fatigue_risk"] == "High"]
    low_util = [s for s in scores if s["utilization"] < 0.2 and s["status"] == "Available"]
    idle_crew = [s for s in scores if s["utilization"] == 0 and s["status"] == "Available"]

    return {
        "crew_scores": scores,
        "top_swaps": swaps,
        "forecast": forecast,
        "alerts": {
            "high_fatigue_count": len(high_fatigue),
            "high_fatigue_crew": [s["name"] for s in high_fatigue],
            "underutilized_count": len(low_util),
            "idle_count": len(idle_crew),
        },
    }
