from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.models import CrewMember, Flight, Role
from data.crew_loader import load_crew
from validators.dgca_validator import (
    check_crew_eligibility, ComplianceResult,
    MAX_DUTY_HOURS, MAX_ROLLING_7_DAY_HOURS,
    MAX_CONSECUTIVE_NIGHT_SHIFTS, MAX_CONSECUTIVE_DAYS_ON,
)
from pathlib import Path

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "crew_standby_list.csv"


def _limit(limits_dict, role, default=0):
    val = limits_dict.get(role, default)
    return val if val != float("inf") else None


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


def get_at_risk_crew(
    csv_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from data.flights_db import get_flights_for_crew

    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)

    violating = []
    critical = []
    warnings = []

    for member in crew:
        checks = []
        member_severity = "ok"

        duty_limit = _limit(MAX_DUTY_HOURS, member.role)
        if duty_limit is not None:
            util = member.current_duty_hours / duty_limit
            if member.current_duty_hours >= duty_limit:
                checks.append({
                    "metric": "Duty Hours",
                    "current": member.current_duty_hours,
                    "limit": duty_limit,
                    "utilization": round(util * 100, 1),
                    "severity": "violating",
                    "detail": f"{member.current_duty_hours:.1f}h / {duty_limit:.0f}h",
                })
                member_severity = "violating"
            elif util >= 0.90:
                checks.append({
                    "metric": "Duty Hours",
                    "current": member.current_duty_hours,
                    "limit": duty_limit,
                    "utilization": round(util * 100, 1),
                    "severity": "critical",
                    "detail": f"{member.current_duty_hours:.1f}h / {duty_limit:.0f}h ({util*100:.0f}%)",
                })
                if member_severity != "violating":
                    member_severity = "critical"
            elif util >= 0.75:
                checks.append({
                    "metric": "Duty Hours",
                    "current": member.current_duty_hours,
                    "limit": duty_limit,
                    "utilization": round(util * 100, 1),
                    "severity": "warning",
                    "detail": f"{member.current_duty_hours:.1f}h / {duty_limit:.0f}h ({util*100:.0f}%)",
                })
                if member_severity == "ok":
                    member_severity = "warning"

        rolling_limit = _limit(MAX_ROLLING_7_DAY_HOURS, member.role)
        if rolling_limit is not None:
            util_r = member.rolling_7_day_hours / rolling_limit
            if member.rolling_7_day_hours >= rolling_limit:
                checks.append({
                    "metric": "Rolling 7-Day Hours",
                    "current": member.rolling_7_day_hours,
                    "limit": rolling_limit,
                    "utilization": round(util_r * 100, 1),
                    "severity": "violating",
                    "detail": f"{member.rolling_7_day_hours:.1f}h / {rolling_limit:.0f}h",
                })
                member_severity = "violating"
            elif util_r >= 0.90:
                checks.append({
                    "metric": "Rolling 7-Day Hours",
                    "current": member.rolling_7_day_hours,
                    "limit": rolling_limit,
                    "utilization": round(util_r * 100, 1),
                    "severity": "critical",
                    "detail": f"{member.rolling_7_day_hours:.1f}h / {rolling_limit:.0f}h ({util_r*100:.0f}%)",
                })
                if member_severity != "violating":
                    member_severity = "critical"
            elif util_r >= 0.75:
                checks.append({
                    "metric": "Rolling 7-Day Hours",
                    "current": member.rolling_7_day_hours,
                    "limit": rolling_limit,
                    "utilization": round(util_r * 100, 1),
                    "severity": "warning",
                    "detail": f"{member.rolling_7_day_hours:.1f}h / {rolling_limit:.0f}h ({util_r*100:.0f}%)",
                })
                if member_severity == "ok":
                    member_severity = "warning"

        night_limit = _limit(MAX_CONSECUTIVE_NIGHT_SHIFTS, member.role, 999)
        if night_limit is not None and night_limit < 999:
            if member.consecutive_night_shifts >= night_limit:
                checks.append({
                    "metric": "Consecutive Night Shifts",
                    "current": member.consecutive_night_shifts,
                    "limit": night_limit,
                    "utilization": 100.0,
                    "severity": "violating",
                    "detail": f"{member.consecutive_night_shifts} / {night_limit}",
                })
                member_severity = "violating"
            elif member.consecutive_night_shifts >= night_limit - 1:
                checks.append({
                    "metric": "Consecutive Night Shifts",
                    "current": member.consecutive_night_shifts,
                    "limit": night_limit,
                    "utilization": round((member.consecutive_night_shifts / night_limit) * 100, 1),
                    "severity": "critical",
                    "detail": f"{member.consecutive_night_shifts} / {night_limit} (one more triggers violation)",
                })
                if member_severity != "violating":
                    member_severity = "critical"

        days_limit = _limit(MAX_CONSECUTIVE_DAYS_ON, member.role, 6)
        if days_limit is not None:
            if member.consecutive_days_on >= days_limit:
                checks.append({
                    "metric": "Consecutive Days On",
                    "current": member.consecutive_days_on,
                    "limit": days_limit,
                    "utilization": 100.0,
                    "severity": "violating",
                    "detail": f"{member.consecutive_days_on} / {days_limit}",
                })
                member_severity = "violating"
            elif member.consecutive_days_on >= days_limit - 1:
                checks.append({
                    "metric": "Consecutive Days On",
                    "current": member.consecutive_days_on,
                    "limit": days_limit,
                    "utilization": round((member.consecutive_days_on / days_limit) * 100, 1),
                    "severity": "critical",
                    "detail": f"{member.consecutive_days_on} / {days_limit} (one more triggers violation)",
                })
                if member_severity != "violating":
                    member_severity = "critical"
            elif member.consecutive_days_on >= days_limit - 2:
                checks.append({
                    "metric": "Consecutive Days On",
                    "current": member.consecutive_days_on,
                    "limit": days_limit,
                    "utilization": round((member.consecutive_days_on / days_limit) * 100, 1),
                    "severity": "warning",
                    "detail": f"{member.consecutive_days_on} / {days_limit}",
                })
                if member_severity == "ok":
                    member_severity = "warning"

        if member.rest_status.lower() != "legal":
            checks.append({
                "metric": "Rest Status",
                "current": member.rest_status,
                "limit": "Legal",
                "utilization": 100.0,
                "severity": "violating",
                "detail": f"Status is '{member.rest_status}'",
            })
            member_severity = "violating"

        if member.days_since_rest >= 5:
            sev = "critical" if member.days_since_rest >= 6 else "warning"
            checks.append({
                "metric": "Days Since Rest",
                "current": member.days_since_rest,
                "limit": 7,
                "utilization": round((member.days_since_rest / 7) * 100, 1),
                "severity": sev,
                "detail": f"{member.days_since_rest} days (weekly rest due soon)" if sev == "warning" else f"{member.days_since_rest} days (weekly rest overdue)",
            })
            if member_severity == "ok" or (sev == "critical" and member_severity != "violating"):
                member_severity = sev

        if member_severity != "ok":
            assigned_flights = get_flights_for_crew(member.crew_id, db_path) if db_path else []
            is_assigned = len(assigned_flights) > 0
            assigned_flight = assigned_flights[0].get("flight_id", "") if assigned_flights else ""
            entry = {
                "crew_id": member.crew_id,
                "name": member.name,
                "role": member.role.value,
                "severity": member_severity,
                "rest_status": member.rest_status,
                "is_assigned": is_assigned,
                "assigned_flight": assigned_flight,
                "checks": checks,
            }
            if member_severity == "violating":
                violating.append(entry)
            elif member_severity == "critical":
                critical.append(entry)
            else:
                warnings.append(entry)

    violating.sort(key=lambda x: x["crew_id"])
    critical.sort(key=lambda x: x["crew_id"])
    warnings.sort(key=lambda x: x["crew_id"])

    return {
        "violating": violating,
        "critical": critical,
        "warnings": warnings,
        "summary": {
            "total_at_risk": len(violating) + len(critical) + len(warnings),
            "violating_count": len(violating),
            "critical_count": len(critical),
            "warning_count": len(warnings),
        },
    }
