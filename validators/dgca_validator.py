from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import inf
from typing import Dict, List, Optional, Tuple

from data.models import CrewMember, Flight, Role


MAX_DUTY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 12.0,
    Role.FO: 12.0,
    Role.CABIN_CREW: 14.0,
    Role.GROUND_STAFF: 10.0,
}

MAX_ROLLING_7_DAY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 35.0,
    Role.FO: 35.0,
    Role.CABIN_CREW: 45.0,
    Role.GROUND_STAFF: inf,
}

MAX_CONSECUTIVE_NIGHT_SHIFTS: Dict[Role, int] = {
    Role.CAPTAIN: 2,
    Role.FO: 2,
    Role.CABIN_CREW: 3,
    Role.GROUND_STAFF: 999,
}

MAX_CONSECUTIVE_DAYS_ON: Dict[Role, int] = {
    Role.CAPTAIN: 6,
    Role.FO: 6,
    Role.CABIN_CREW: 6,
    Role.GROUND_STAFF: 6,
}

STANDARD_DUTY_THRESHOLD: Dict[Role, float] = {
    Role.CAPTAIN: 10.0,
    Role.FO: 10.0,
    Role.CABIN_CREW: 10.0,
    Role.GROUND_STAFF: 8.0,
}

MIN_REST_HOURS: float = 10.0
MIN_LAYOVER_MINUTES: int = 600


@dataclass
class ComplianceResult:
    eligible: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "eligible": self.eligible,
            "violations": self.violations,
            "warnings": self.warnings,
        }


def _limit(mapping: Dict, role: Role, default: float = inf) -> float:
    return mapping.get(role, default)


def check_crew_eligibility(
    member: CrewMember,
    flights: Optional[List[Flight]] = None,
    scenario_flight_hours: float = 0.0,
    scenario_is_night_duty: bool = False,
) -> ComplianceResult:
    violations: List[str] = []
    warnings: List[str] = []

    if member.rest_status != "Legal":
        violations.append(f"Rest status is '{member.rest_status}' (must be 'Legal')")

    max_duty = _limit(MAX_DUTY_HOURS, member.role)
    projected_duty = member.current_duty_hours + scenario_flight_hours
    if projected_duty > max_duty:
        violations.append(
            f"Projected duty hours ({projected_duty:.1f}) exceed max ({max_duty:.0f}h)"
        )

    rolling_limit = _limit(MAX_ROLLING_7_DAY_HOURS, member.role)
    if rolling_limit != inf:
        projected_rolling = member.rolling_7_day_hours + scenario_flight_hours
        if projected_rolling > rolling_limit:
            violations.append(
                f"Rolling 7-day hours ({projected_rolling:.1f}) exceed max ({rolling_limit:.0f}h)"
            )

    if scenario_is_night_duty:
        night_limit = _limit(MAX_CONSECUTIVE_NIGHT_SHIFTS, member.role, 999)
        if member.consecutive_night_shifts >= night_limit:
            violations.append(
                f"Consecutive night duties ({member.consecutive_night_shifts}) would reach limit ({night_limit})"
            )

    days_limit = _limit(MAX_CONSECUTIVE_DAYS_ON, member.role, 6)
    if member.consecutive_days_on >= days_limit:
        warnings.append(
            f"Consecutive days on duty ({member.consecutive_days_on}) is at threshold ({days_limit})"
        )

    if flights:
        for flight in flights:
            if not member.is_rated_on(flight.aircraft_type):
                violations.append(
                    f"Not type-rated for {flight.aircraft_type} (flight {flight.flight_id})"
                )
            if member.base_airport and member.base_airport != flight.origin:
                if not member.base_airport:
                    warnings.append(f"No base airport assigned")
                else:
                    warnings.append(
                        f"Base airport ({member.base_airport}) differs from flight origin ({flight.origin})"
                    )

    return ComplianceResult(
        eligible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_assignment_legality(
    assignments: Dict[str, List[str]],
    crew_map: Dict[str, CrewMember],
    flights: List[Flight],
) -> Dict[str, ComplianceResult]:
    results: Dict[str, ComplianceResult] = {}
    for crew_id, assigned_flight_ids in assignments.items():
        member = crew_map.get(crew_id)
        if not member:
            results[crew_id] = ComplianceResult(
                eligible=False,
                violations=[f"Crew member {crew_id} not found in roster"],
            )
            continue
        assigned_flights = [f for f in flights if f.flight_id in assigned_flight_ids]
        total_hours = sum(f.flight_hours for f in assigned_flights)
        is_night = any(f.is_night_duty for f in assigned_flights)
        results[crew_id] = check_crew_eligibility(
            member, assigned_flights, total_hours, is_night
        )
    return results


def compute_cost(
    member: CrewMember,
    scenario_flight_hours: float,
) -> float:
    threshold = _limit(STANDARD_DUTY_THRESHOLD, member.role, 10.0)
    regular_remaining = max(threshold - member.current_duty_hours, 0.0)
    regular_hours = min(scenario_flight_hours, regular_remaining)
    overtime_hours = max(scenario_flight_hours - regular_remaining, 0.0)
    return (member.base_cost * regular_hours) + (
        member.base_cost * member.overtime_multiplier * overtime_hours
    )
