from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import inf
from typing import Dict, List, Optional, Tuple

from data.models import CrewMember, Flight, Role


# ---------------------------------------------------------------------------
# Per-role limits (DGCA CAR Section 7 Series J Part III)
# ---------------------------------------------------------------------------
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

# WOCL (Window of circadian low) — 02:00 to 06:00 local time
WOCL_START_HOUR = 2
WOCL_END_HOUR = 6

# Cumulative hour limits (per role)
MAX_14_DAY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 60.0,
    Role.FO: 60.0,
    Role.CABIN_CREW: 80.0,
    Role.GROUND_STAFF: inf,
}

MAX_28_DAY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 100.0,
    Role.FO: 100.0,
    Role.CABIN_CREW: 130.0,
    Role.GROUND_STAFF: inf,
}

MAX_90_DAY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 270.0,
    Role.FO: 270.0,
    Role.CABIN_CREW: 350.0,
    Role.GROUND_STAFF: inf,
}

MAX_365_DAY_HOURS: Dict[Role, float] = {
    Role.CAPTAIN: 1000.0,
    Role.FO: 1000.0,
    Role.CABIN_CREW: 1300.0,
    Role.GROUND_STAFF: inf,
}

# Weekly rest requirements
MIN_WEEKLY_REST_HOURS = 48.0
MIN_WEEKLY_REST_LOCAL_NIGHTS = 2
MIN_WEEKLY_REST_3_NIGHT_DUTIES_HOURS = 60.0

# Augmented crew provisions
AUGMENTED_CREW_3_THRESHOLD_HOURS = 8.0
AUGMENTED_CREW_4_THRESHOLD_HOURS = 10.0

# ULR (Ultra Long Range)
ULR_THRESHOLD_HOURS = 8.0
ULR_MIN_CREW = 4
ULR_POST_REST_HOURS = 120.0

# Unforeseen operational circumstance extensions
UNFORESEEN_EXTENSION_30_MIN = 30
UNFORESEEN_EXTENSION_60_MIN = 60

# Split duty
MAX_SPLIT_DUTY_EXTENSION_HOURS = 2.0
MIN_SPLIT_REST_HOURS = 3.0


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


def _is_night_time(hour: int) -> bool:
    return hour >= WOCL_START_HOUR and hour < WOCL_END_HOUR


def _check_wocl_violation(member: CrewMember, flight: Flight) -> Optional[str]:
    dep_hour = flight.std.hour
    arr_hour = (flight.sta).hour
    duty_start_hour = dep_hour - 1

    if duty_start_hour < WOCL_END_HOUR and arr_hour >= WOCL_START_HOUR:
        if duty_start_hour < WOCL_START_HOUR and arr_hour >= WOCL_END_HOUR:
            return (
                f"WOCL violation: duty spans entire WOCL window "
                f"({WOCL_START_HOUR:02d}:00-{WOCL_END_HOUR:02d}:00) "
                f"(dep {dep_hour:02d}, arr {arr_hour:02d})"
            )
        if _is_night_time(duty_start_hour) or _is_night_time(arr_hour):
            return (
                f"WOCL boundary crossing: duty starts/ends within WOCL "
                f"({WOCL_START_HOUR:02d}:00-{WOCL_END_HOUR:02d}:00)"
            )
    return None


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


def check_cumulative_limits(member: CrewMember) -> ComplianceResult:
    violations: List[str] = []
    warnings: List[str] = []

    h14_limit = _limit(MAX_14_DAY_HOURS, member.role)
    if h14_limit != inf and member.rolling_7_day_hours * 2 > h14_limit:
        warnings.append(
            f"14-day cumulative hours trending high: "
            f"~{member.rolling_7_day_hours * 2:.0f}h (limit {h14_limit:.0f}h)"
        )

    h28_limit = _limit(MAX_28_DAY_HOURS, member.role)
    if h28_limit != inf and member.hours_flown_30_days > h28_limit:
        violations.append(
            f"28-day cumulative hours ({member.hours_flown_30_days:.1f}) exceed max ({h28_limit:.0f}h)"
        )

    h90_limit = _limit(MAX_90_DAY_HOURS, member.role)
    if h90_limit != inf and member.hours_flown_30_days * 3 > h90_limit:
        warnings.append(
            f"90-day cumulative hours trending high: "
            f"~{member.hours_flown_30_days * 3:.0f}h (limit {h90_limit:.0f}h)"
        )

    return ComplianceResult(
        eligible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_weekly_rest(member: CrewMember) -> ComplianceResult:
    warnings: List[str] = []

    if member.days_since_rest < 7:
        days_to_rest = 7 - member.days_since_rest
        warnings.append(
            f"Must take weekly rest within {days_to_rest} day(s) "
            f"(last rest was {member.days_since_rest} days ago)"
        )

    if member.consecutive_night_shifts >= 3:
        warnings.append(
            f"3+ night duties in 7 days: weekly rest must be {MIN_WEEKLY_REST_3_NIGHT_DUTIES_HOURS:.0f}h "
            f"(standard {MIN_WEEKLY_REST_HOURS:.0f}h)"
        )

    return ComplianceResult(
        eligible=True,
        violations=[],
        warnings=warnings,
    )


def check_augmented_crew(flights: List[Flight], crew_count: int) -> ComplianceResult:
    warnings: List[str] = []

    for flight in flights:
        if flight.flight_hours >= AUGMENTED_CREW_4_THRESHOLD_HOURS:
            if crew_count < 4:
                warnings.append(
                    f"Flight {flight.flight_id} ({flight.flight_hours:.1f}h): "
                    f"recommended 4-crew augmented provision (have {crew_count})"
                )
        elif flight.flight_hours >= AUGMENTED_CREW_3_THRESHOLD_HOURS:
            if crew_count < 3:
                warnings.append(
                    f"Flight {flight.flight_id} ({flight.flight_hours:.1f}h): "
                    f"recommended 3-crew augmented provision (have {crew_count})"
                )

    return ComplianceResult(
        eligible=True,
        violations=[],
        warnings=warnings,
    )


def check_ulr_compliance(member: CrewMember, flights: List[Flight]) -> ComplianceResult:
    violations: List[str] = []
    warnings: List[str] = []

    for flight in flights:
        if flight.flight_hours >= ULR_THRESHOLD_HOURS:
            if member.consecutive_days_on >= 1:
                post_rest = member.days_since_rest * 24
                if post_rest < ULR_POST_REST_HOURS:
                    violations.append(
                        f"ULR flight {flight.flight_id} ({flight.flight_hours:.1f}h): "
                        f"post-ULR rest {post_rest:.0f}h < required {ULR_POST_REST_HOURS:.0f}h"
                    )

            if not member.is_rated_on(flight.aircraft_type):
                violations.append(
                    f"ULR flight {flight.flight_id}: not type-rated for {flight.aircraft_type}"
                )

    return ComplianceResult(
        eligible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_wocl(member: CrewMember, flights: List[Flight]) -> ComplianceResult:
    violations: List[str] = []
    warnings: List[str] = []

    for flight in flights:
        wocl_issue = _check_wocl_violation(member, flight)
        if wocl_issue:
            violations.append(wocl_issue)

        dep_hour = flight.std.hour
        if _is_night_time(dep_hour):
            warnings.append(
                f"Flight {flight.flight_id} departs during WOCL ({dep_hour:02d}:00)"
            )

    return ComplianceResult(
        eligible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_split_duty(
    member: CrewMember,
    duty1_hours: float,
    rest_hours: float,
    duty2_hours: float,
) -> ComplianceResult:
    violations: List[str] = []
    warnings: List[str] = []

    if rest_hours < MIN_SPLIT_REST_HOURS:
        violations.append(
            f"Split duty rest ({rest_hours:.1f}h) below minimum ({MIN_SPLIT_REST_HOURS:.0f}h)"
        )

    total_duty = duty1_hours + duty2_hours
    max_duty = _limit(MAX_DUTY_HOURS, member.role) + MAX_SPLIT_DUTY_EXTENSION_HOURS

    if total_duty > max_duty:
        violations.append(
            f"Split duty total ({total_duty:.1f}h) exceeds extended max ({max_duty:.0f}h)"
        )

    if duty1_hours > _limit(MAX_DUTY_HOURS, member.role):
        warnings.append(
            f"First duty period ({duty1_hours:.1f}h) exceeds standard max "
            f"({MAX_DUTY_HOURS.get(member.role, 12):.0f}h) — split duty extension required"
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


def full_compliance_check(
    member: CrewMember,
    flights: Optional[List[Flight]] = None,
    scenario_flight_hours: float = 0.0,
    scenario_is_night_duty: bool = False,
) -> ComplianceResult:
    all_violations: List[str] = []
    all_warnings: List[str] = []

    basic = check_crew_eligibility(member, flights, scenario_flight_hours, scenario_is_night_duty)
    all_violations.extend(basic.violations)
    all_warnings.extend(basic.warnings)

    cumulative = check_cumulative_limits(member)
    all_violations.extend(cumulative.violations)
    all_warnings.extend(cumulative.warnings)

    weekly = check_weekly_rest(member)
    all_violations.extend(weekly.violations)
    all_warnings.extend(weekly.warnings)

    if flights:
        wocl = check_wocl(member, flights)
        all_violations.extend(wocl.violations)
        all_warnings.extend(wocl.warnings)

        ulr = check_ulr_compliance(member, flights)
        all_violations.extend(ulr.violations)
        all_warnings.extend(ulr.warnings)

    return ComplianceResult(
        eligible=len(all_violations) == 0,
        violations=all_violations,
        warnings=all_warnings,
    )
