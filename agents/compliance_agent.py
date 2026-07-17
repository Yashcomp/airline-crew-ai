from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.models import CrewMember, Flight, Role
from data.crew_loader import load_crew
from validators.dgca_validator import check_crew_eligibility, ComplianceResult
from pathlib import Path

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "crew_standby_list.csv"


def validate_single_crew(
    crew_id: str,
    flight_ids: Optional[List[str]] = None,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    member = next((m for m in crew if m.crew_id == crew_id), None)
    if not member:
        return {"error": f"Crew member {crew_id} not found"}

    result = check_crew_eligibility(member)
    return {
        "crew_id": member.crew_id,
        "name": member.name,
        "role": member.role.value,
        "rest_status": member.rest_status,
        "current_duty_hours": member.current_duty_hours,
        "rolling_7_day_hours": member.rolling_7_day_hours,
        "compliance": result.to_dict(),
    }


def batch_validate(
    csv_path: Optional[str] = None,
    scenario_flight_hours: float = 0.0,
    scenario_is_night_duty: bool = False,
) -> Dict[str, Any]:
    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    results = {}
    eligible_count = 0
    for member in crew:
        result = check_crew_eligibility(
            member,
            scenario_flight_hours=scenario_flight_hours,
            scenario_is_night_duty=scenario_is_night_duty,
        )
        results[member.crew_id] = {
            "name": member.name,
            "role": member.role.value,
            "eligible": result.eligible,
            "violations": result.violations,
            "warnings": result.warnings,
        }
        if result.eligible:
            eligible_count += 1

    return {
        "total": len(crew),
        "eligible": eligible_count,
        "ineligible": len(crew) - eligible_count,
        "details": results,
    }
