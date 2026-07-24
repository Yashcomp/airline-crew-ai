from __future__ import annotations

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.crew_loader import load_crew
from data.flights_db import (
    assign_crew_to_flight, get_crew_for_flight, get_flights_for_crew,
    is_crew_assigned, get_standby_crew, get_flight,
)
from data.models import CrewMember, Flight, Role
from validators.dgca_validator import check_crew_eligibility


REQUIRED_CREW = {
    "Captain": 1,
    "FO": 1,
    "CabinCrew": 4,
    "GroundStaff": 2,
}


def auto_assign_flight(
    flight_id: str,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    crew = load_crew(csv_path)
    already_assigned = get_crew_for_flight(flight_id, db_path)
    assigned_ids = {a["crew_id"] for a in already_assigned}

    new_assignments = []
    errors = []

    for role_name, count in REQUIRED_CREW.items():
        needed = count
        for member in crew:
            if needed <= 0:
                break
            if member.crew_id in assigned_ids:
                continue
            if is_crew_assigned(member.crew_id, db_path):
                continue
            if member.rest_status.lower() != "legal":
                continue

            eligible = check_crew_eligibility(
                member,
                flights=[flight],
                scenario_flight_hours=flight.flight_hours,
                scenario_is_night_duty=flight.is_night_duty,
            )
            if not eligible.eligible:
                continue

            role_map = {
                "Captain": Role.CAPTAIN,
                "FO": Role.FO,
                "CabinCrew": Role.CABIN_CREW,
                "GroundStaff": Role.GROUND_STAFF,
            }
            if member.role != role_map.get(role_name):
                continue

            result = assign_crew_to_flight(member.crew_id, flight_id, role_name, db_path)
            if result["status"] == "success":
                new_assignments.append({
                    "crew_id": member.crew_id,
                    "name": member.name,
                    "role": role_name,
                })
                assigned_ids.add(member.crew_id)
                needed -= 1
            else:
                errors.append(result.get("message", "Assignment failed"))

    return {
        "status": "success",
        "flight_id": flight_id,
        "assigned_count": len(new_assignments),
        "assignments": new_assignments,
        "errors": errors,
        "already_assigned": len(already_assigned),
    }


def create_staff(
    csv_path: str,
    crew_id: str,
    name: str,
    role: str,
    base_airport: str = "DEL",
    qualifications: str = "",
    base_cost: float = 1.0,
) -> Dict[str, Any]:
    path = Path(csv_path)
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            existing = list(reader)

    for row in existing:
        if row.get("crew_id", "").upper() == crew_id.upper():
            return {"status": "error", "message": f"Crew ID {crew_id} already exists."}

    new_row = {
        "crew_id": crew_id.upper(),
        "name": name,
        "role": role,
        "current_duty_hours": "0",
        "rolling_7_day_hours": "0",
        "consecutive_night_shifts": "0",
        "rest_status": "Legal",
        "base_cost": str(base_cost),
        "overtime_multiplier": "1.0",
        "qualifications": qualifications,
        "base_airport": base_airport.upper(),
        "seniority": "0",
        "hours_flown_30_days": "0",
        "days_since_rest": "0",
        "consecutive_days_on": "0",
    }

    fieldnames = [
        "crew_id", "name", "role", "current_duty_hours", "rolling_7_day_hours",
        "consecutive_night_shifts", "rest_status", "base_cost", "overtime_multiplier",
        "qualifications", "base_airport", "seniority", "hours_flown_30_days",
        "days_since_rest", "consecutive_days_on",
    ]

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(new_row)

    return {
        "status": "success",
        "message": f"Staff {name} ({crew_id.upper()}) created successfully.",
        "crew_id": crew_id.upper(),
    }


def get_assignment_summary(
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    crew = load_crew(csv_path)
    standby = get_standby_crew(csv_path, db_path)
    assigned_count = len(crew) - len(standby)

    assigned_crew = []
    for member in crew:
        if not is_crew_assigned(member.crew_id, db_path):
            continue
        flights = get_flights_for_crew(member.crew_id, db_path)
        assigned_crew.append({
            "crew_id": member.crew_id,
            "name": member.name,
            "role": member.role.value,
            "assigned_flights": [f["flight_id"] for f in flights],
            "flight_count": len(flights),
        })

    return {
        "total_crew": len(crew),
        "assigned_count": assigned_count,
        "standby_count": len(standby),
        "assigned_crew": assigned_crew,
        "standby_crew": standby,
    }
