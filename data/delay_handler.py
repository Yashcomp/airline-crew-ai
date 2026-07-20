from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.crew_loader import load_crew
from data.flights_db import (
    get_flight, update_flight_status, get_crew_for_flight,
    unassign_crew_from_flight, get_flight_stats,
)
from data.models import Flight, Role
from validators.dgca_validator import check_crew_eligibility, MAX_DUTY_HOURS, MAX_ROLLING_7_DAY_HOURS, _limit


def _update_crew_duty_hours(csv_path: str, crew_id: str, extra_hours: float) -> None:
    path = Path(csv_path)
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get("crew_id", "").upper() == crew_id.upper():
                current = float(row.get("current_duty_hours", 0))
                rolling = float(row.get("rolling_7_day_hours", 0))
                row["current_duty_hours"] = str(round(current + extra_hours, 1))
                row["rolling_7_day_hours"] = str(round(rolling + extra_hours, 1))
            rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_delay(
    flight_id: str,
    delay_minutes: int,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    update_flight_status(flight_id, "delayed", db_path=db_path)

    delay_hours = delay_minutes / 60.0
    assigned = get_crew_for_flight(flight_id, db_path)
    if not assigned:
        return {
            "status": "success",
            "flight_id": flight_id,
            "delay_minutes": delay_minutes,
            "crew_impact": [],
            "message": "Flight delayed. No crew was assigned.",
        }

    all_crew = load_crew(csv_path)
    crew_map = {m.crew_id: m for m in all_crew}

    crew_impact = []
    unassigned_crew = []

    for assignment in assigned:
        cid = assignment["crew_id"]
        member = crew_map.get(cid)
        if not member:
            continue

        projected_duty = member.current_duty_hours + flight.flight_hours + delay_hours
        projected_rolling = member.rolling_7_day_hours + flight.flight_hours + delay_hours
        max_duty = _limit(MAX_DUTY_HOURS, member.role)
        max_rolling = _limit(MAX_ROLLING_7_DAY_HOURS, member.role)

        violations = []
        if projected_duty > max_duty:
            violations.append(f"Duty {projected_duty:.1f}h > {max_duty:.0f}h limit")
        if max_rolling != float("inf") and projected_rolling > max_rolling:
            violations.append(f"Rolling {projected_rolling:.1f}h > {max_rolling:.0f}h limit")

        has_violation = len(violations) > 0
        if has_violation:
            unassign_crew_from_flight(cid, flight_id, db_path)
            _update_crew_duty_hours(csv_path, cid, flight.flight_hours + delay_hours)
            unassigned_crew.append({
                "crew_id": cid,
                "name": member.name,
                "role": member.role.value,
                "reason": "; ".join(violations),
            })

        crew_impact.append({
            "crew_id": cid,
            "name": member.name,
            "role": member.role.value,
            "current_duty": member.current_duty_hours,
            "projected_duty": round(projected_duty, 1),
            "duty_limit": max_duty,
            "status": "unassigned" if has_violation else "ok",
            "violations": violations,
        })

    return {
        "status": "success",
        "flight_id": flight_id,
        "delay_minutes": delay_minutes,
        "delay_hours": delay_hours,
        "crew_impact": crew_impact,
        "unassigned_count": len(unassigned_crew),
        "unassigned_crew": unassigned_crew,
    }


def process_cancellation(
    flight_id: str,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    update_flight_status(flight_id, "cancelled", db_path=db_path)

    assigned = get_crew_for_flight(flight_id, db_path)
    freed_crew = []
    for assignment in assigned:
        unassign_crew_from_flight(assignment["crew_id"], flight_id, db_path)
        freed_crew.append({
            "crew_id": assignment["crew_id"],
            "role": assignment["role"],
        })

    return {
        "status": "success",
        "flight_id": flight_id,
        "freed_count": len(freed_crew),
        "freed_crew": freed_crew,
    }


def find_replacement_crew(
    flight_id: str,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from data.staff_manager import REQUIRED_CREW

    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    assigned = get_crew_for_flight(flight_id, db_path)
    assigned_role_counts = {}
    for a in assigned:
        role = a["role"]
        assigned_role_counts[role] = assigned_role_counts.get(role, 0) + 1

    missing = {}
    for role_name, required_count in REQUIRED_CREW.items():
        have = assigned_role_counts.get(role_name, 0)
        if have < required_count:
            missing[role_name] = required_count - have

    if not missing:
        return {
            "status": "success",
            "flight_id": flight_id,
            "message": "All roles are fully staffed.",
            "candidates": {},
        }

    all_crew = load_crew(csv_path)
    role_map = {
        "Captain": Role.CAPTAIN,
        "FO": Role.FO,
        "CabinCrew": Role.CABIN_CREW,
        "GroundStaff": Role.GROUND_STAFF,
    }

    assigned_ids = {a["crew_id"] for a in assigned}

    candidates = {}
    for role_name, needed in missing.items():
        target_role = role_map.get(role_name)
        role_candidates = []
        for member in all_crew:
            if member.crew_id in assigned_ids:
                continue
            if member.rest_status.lower() != "legal":
                continue
            if member.role != target_role:
                continue

            eligible = check_crew_eligibility(
                member,
                flights=[flight],
                scenario_flight_hours=flight.flight_hours,
                scenario_is_night_duty=flight.is_night_duty,
            )
            cost = member.base_cost * member.overtime_multiplier * flight.flight_hours
            role_candidates.append({
                "crew_id": member.crew_id,
                "name": member.name,
                "role": role_name,
                "base_airport": member.base_airport,
                "current_duty": member.current_duty_hours,
                "rolling_7_day": member.rolling_7_day_hours,
                "cost": round(cost, 2),
                "eligible": eligible.eligible,
                "violations": eligible.violations,
                "warnings": eligible.warnings,
            })

        role_candidates.sort(key=lambda x: (not x["eligible"], x["cost"]))
        candidates[role_name] = {
            "needed": needed,
            "candidates": role_candidates,
        }

    return {
        "status": "success",
        "flight_id": flight_id,
        "candidates": candidates,
    }
