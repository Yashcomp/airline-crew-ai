from __future__ import annotations

import json
import os
import re
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
    use_llm: bool = False,
    delay_hours: float = 0.0,
) -> ComplianceResult:
    if use_llm and flights:
        return get_llm_compliance_decision(member, flights[0], delay_hours)

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
            if member.role != Role.GROUND_STAFF and not member.is_rated_on(flight.aircraft_type):
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


def _build_crew_summary(member: CrewMember) -> str:
    quals = ", ".join(q.aircraft_type for q in member.qualifications)
    return (
        f"Crew ID: {member.crew_id}\n"
        f"Name: {member.name}\n"
        f"Role: {member.role.value}\n"
        f"Rest Status: {member.rest_status}\n"
        f"Current Duty Hours: {member.current_duty_hours}\n"
        f"Rolling 7-Day Hours: {member.rolling_7_day_hours}\n"
        f"Consecutive Night Shifts: {member.consecutive_night_shifts}\n"
        f"Consecutive Days On: {member.consecutive_days_on}\n"
        f"Days Since Rest: {member.days_since_rest}\n"
        f"Hours Flown (30 days): {member.hours_flown_30_days}\n"
        f"Base Airport: {member.base_airport}\n"
        f"Type Ratings: {quals}\n"
    )


def _build_flight_summary(flight: Flight, delay_hours: float = 0.0) -> str:
    return (
        f"Flight ID: {flight.flight_id}\n"
        f"Route: {flight.origin} → {flight.destination}\n"
        f"Aircraft: {flight.aircraft_type}\n"
        f"Flight Duration: {flight.flight_hours}h\n"
        f"Scheduled Departure: {flight.std.strftime('%H:%M')}\n"
        f"Is Night Duty: {flight.is_night_duty}\n"
        f"Delay: {delay_hours}h\n"
    )


def _build_rules_context() -> str:
    hardcoded_rules = (
        "DGCA CAR Section 7 Series J Part III - OFFICIAL DUTY LIMITS:\n\n"
        "MAXIMUM DUTY HOURS (per 24 hours):\n"
        "- Captain: 12 hours\n"
        "- First Officer (FO): 12 hours\n"
        "- Cabin Crew: 14 hours\n"
        "- Ground Staff: 10 hours\n\n"
        "ROLLING 7-DAY HOUR LIMITS:\n"
        "- Captain: 35 hours\n"
        "- First Officer (FO): 35 hours\n"
        "- Cabin Crew: 45 hours\n"
        "- Ground Staff: No limit\n\n"
        "CONSECUTIVE NIGHT SHIFT LIMITS:\n"
        "- Captain: 2 consecutive nights\n"
        "- First Officer (FO): 2 consecutive nights\n"
        "- Cabin Crew: 3 consecutive nights\n"
        "- Ground Staff: No limit\n\n"
        "REST REQUIREMENTS:\n"
        "- Minimum rest between duty periods: 10 hours\n"
        "- Rest must include a local night (if preceding duty > 18 hours)\n\n"
        "WOCL (Window of Circadian Low):\n"
        "- Hours: 02:00 to 06:00 local time\n"
        "- Duty spanning entire WOCL is prohibited\n"
        "- Ground Staff are EXEMPT from WOCL restrictions\n\n"
        "TYPE RATING:\n"
        "- Crew must be type-rated for the aircraft they operate\n"
        "- GROUND STAFF ARE EXEMPT from type-rating checks\n\n"
        "UNFORESEEN CIRCUMSTANCES:\n"
        "- 30-minute extension allowed for unforeseen operational circumstances\n"
        "- 60-minute extension with commander approval\n\n"
        "SPLIT DUTY:\n"
        "- Maximum 2-hour extension allowed\n"
        "- Minimum 3-hour rest between duty periods\n\n"
        "DELAY/REPLACEMENT RULES:\n"
        "- Standby/replacement crew duty starts when the flight DEPARTS\n"
        "- Do NOT add delay hours to standby crew's projected duty\n"
        "- Evaluate standby crew against flight duration only\n"
    )

    try:
        from rag_engine import retrieve_legal_guidance
        rag_query = "DGCA flight duty period definition duty period limits"
        rag_context = retrieve_legal_guidance(rag_query)
        if rag_context and len(rag_context) > 50:
            return hardcoded_rules + "\n\nADDITIONAL DGCA CONTEXT:\n" + rag_context
    except Exception:
        pass

    return hardcoded_rules


def get_llm_compliance_decision(
    member: CrewMember,
    flight: Flight,
    delay_hours: float = 0.0,
) -> ComplianceResult:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
    except ImportError:
        return check_crew_eligibility(
            member,
            flights=[flight],
            scenario_flight_hours=flight.flight_hours,
            scenario_is_night_duty=flight.is_night_duty,
        )

    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        return check_crew_eligibility(
            member,
            flights=[flight],
            scenario_flight_hours=flight.flight_hours,
            scenario_is_night_duty=flight.is_night_duty,
        )

    crew_summary = _build_crew_summary(member)
    flight_summary = _build_flight_summary(flight, delay_hours)

    system_prompt = """You are a DGCA compliance officer. Your job is to VALIDATE whether a crew member can be assigned to a flight.

STEP-BY-STEP VALIDATION:
1. Check REST STATUS: Must be "Legal"
2. Check DUTY HOURS: projected_duty = current_duty + flight_duration. Must be <= role limit.
3. Check ROLLING 7-DAY: projected_rolling = rolling_7day + flight_duration. Must be <= role limit.
4. Check TYPE RATING: Must have rating for aircraft type (Ground Staff EXEMPT).
5. Check WOCL: If flight departs between 02:00-06:00, flag as night duty concern.
6. Check CONSECUTIVE NIGHTS: Must not exceed role limit.

ROLE LIMITS:
- Captain/FO: 12h duty, 35h rolling, 2 consecutive nights
- Cabin Crew: 14h duty, 45h rolling, 3 consecutive nights  
- Ground Staff: 10h duty, no rolling limit, no night limit

IMPORTANT: 
- For STANDBY/REPLACEMENT crew, duty starts when flight departs, NOT when delay is announced
- Do NOT add delay_hours to the calculation
- Ground Staff are EXEMPT from type-rating checks

RESPOND IN THIS EXACT JSON FORMAT:
{
    "eligible": true or false,
    "violations": ["specific rule violations only if they exist"],
    "warnings": ["minor concerns only"],
    "reasoning": "brief explanation with numbers"
}"""

    projected_duty = member.current_duty_hours + flight.flight_hours
    projected_rolling = member.rolling_7_day_hours + flight.flight_hours

    human_prompt = f"""VALIDATE THIS CREW MEMBER:

CREW: {member.name} ({member.role.value})
- Rest Status: {member.rest_status}
- Current Duty: {member.current_duty_hours}h
- Rolling 7-day: {member.rolling_7_day_hours}h
- Consecutive Nights: {member.consecutive_night_shifts}
- Type Ratings: {', '.join(q.aircraft_type for q in member.qualifications)}

FLIGHT: {flight.flight_id} ({flight.aircraft_type})
- Duration: {flight.flight_hours}h
- Departs: {flight.std.strftime('%H:%M')}
- Night Duty: {flight.is_night_duty}

CALCULATED VALUES:
- Projected Duty: {member.current_duty_hours}h + {flight.flight_hours}h = {projected_duty:.1f}h
- Projected Rolling: {member.rolling_7_day_hours}h + {flight.flight_hours}h = {projected_rolling:.1f}h

VALIDATE: Is {projected_duty:.1f}h <= {MAX_DUTY_HOURS.get(member.role, 12)}h limit? Is {projected_rolling:.1f}h <= {MAX_ROLLING_7_DAY_HOURS.get(member.role, 35)}h limit?

Respond with JSON decision."""

    try:
        model = ChatOpenAI(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            temperature=0.0,
            request_timeout=10,
        )
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])

        response_text = response.content
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            payload = json.loads(json_match.group())
        else:
            payload = json.loads(response_text)

        eligible = payload.get("eligible", False)
        violations = payload.get("violations", [])
        warnings = payload.get("warnings", [])

        if not eligible and not violations:
            violations = ["LLM determined crew member is not eligible (no specific violation provided)"]

        return ComplianceResult(
            eligible=eligible,
            violations=violations,
            warnings=warnings,
        )

    except Exception as e:
        return ComplianceResult(
            eligible=False,
            violations=[f"LLM decision failed: {str(e)[:100]}"],
            warnings=["Falling back to hardcoded rules"],
        )


def full_compliance_check(
    member: CrewMember,
    flights: Optional[List[Flight]] = None,
    scenario_flight_hours: float = 0.0,
    scenario_is_night_duty: bool = False,
    use_llm: bool = False,
    delay_hours: float = 0.0,
) -> ComplianceResult:
    if use_llm and flights:
        return get_llm_compliance_decision(member, flights[0], delay_hours)

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
