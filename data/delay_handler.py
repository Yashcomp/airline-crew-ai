from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.crew_loader import load_crew
from data.flights_db import (
    get_flight, update_flight_status, get_crew_for_flight,
    unassign_crew_from_flight, get_flight_stats, assign_crew_to_flight,
    is_crew_assigned,
)
from data.models import Flight, Role
from validators.dgca_validator import (
    check_crew_eligibility, full_compliance_check, compute_cost,
    MAX_DUTY_HOURS, MAX_ROLLING_7_DAY_HOURS, _limit,
)


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
            "delay_hours": delay_hours,
            "crew_impact": [],
            "unassigned_count": 0,
            "unassigned_crew": [],
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


def analyze_delay_impact(
    flight_id: str,
    delay_minutes: int,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    delay_hours = delay_minutes / 60.0
    assigned = get_crew_for_flight(flight_id, db_path)
    if not assigned:
        return {
            "status": "success",
            "flight_id": flight_id,
            "delay_minutes": delay_minutes,
            "delay_hours": delay_hours,
            "flight_info": {
                "origin": flight.origin,
                "destination": flight.destination,
                "aircraft_type": flight.aircraft_type,
                "flight_hours": flight.flight_hours,
            },
            "assigned_crew": [],
            "ineligible_crew": [],
            "message": "No crew assigned to this flight.",
        }

    all_crew = load_crew(csv_path)
    crew_map = {m.crew_id: m for m in all_crew}

    assigned_crew = []
    ineligible_crew = []

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

        is_ineligible = len(violations) > 0
        entry = {
            "crew_id": cid,
            "name": member.name,
            "role": member.role.value,
            "current_duty": member.current_duty_hours,
            "projected_duty": round(projected_duty, 1),
            "duty_limit": max_duty,
            "violations": violations,
        }

        assigned_crew.append(entry)
        if is_ineligible:
            ineligible_crew.append(entry)

    return {
        "status": "success",
        "flight_id": flight_id,
        "delay_minutes": delay_minutes,
        "delay_hours": delay_hours,
        "flight_info": {
            "origin": flight.origin,
            "destination": flight.destination,
            "aircraft_type": flight.aircraft_type,
            "flight_hours": flight.flight_hours,
        },
        "assigned_crew": assigned_crew,
        "ineligible_crew": ineligible_crew,
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
    use_llm: bool = False,
) -> Dict[str, Any]:
    flight = get_flight(flight_id, db_path)
    if not flight:
        return {"status": "error", "message": f"Flight {flight_id} not found."}

    all_crew = load_crew(csv_path)
    assigned = get_crew_for_flight(flight_id, db_path)
    assigned_ids = {a["crew_id"] for a in assigned}

    scenario_hours = flight.flight_hours

    assigned_crew_details = []
    eligible_standby = []
    eligible_assigned_elsewhere = []
    ineligible = []

    for member in all_crew:
        basic = full_compliance_check(
            member,
            flights=[flight],
            scenario_flight_hours=scenario_hours,
            scenario_is_night_duty=flight.is_night_duty,
            use_llm=False,
        )

        if use_llm and basic.eligible:
            result = full_compliance_check(
                member,
                flights=[flight],
                scenario_flight_hours=scenario_hours,
                scenario_is_night_duty=flight.is_night_duty,
                use_llm=True,
            )
        else:
            result = basic
        cost = round(compute_cost(member, scenario_hours), 2)
        is_on_this_flight = member.crew_id in assigned_ids
        is_busy_elsewhere = is_crew_assigned(member.crew_id, db_path) and not is_on_this_flight

        entry = {
            "crew_id": member.crew_id,
            "name": member.name,
            "role": member.role.value,
            "base_airport": member.base_airport,
            "current_duty_hours": member.current_duty_hours,
            "rolling_7_day_hours": member.rolling_7_day_hours,
            "consecutive_night_shifts": member.consecutive_night_shifts,
            "rest_status": member.rest_status,
            "qualifications": [q.aircraft_type for q in member.qualifications],
            "eligible": result.eligible,
            "violations": result.violations,
            "warnings": result.warnings,
            "cost": cost,
        }

        if is_on_this_flight:
            if not result.eligible:
                entry["status_label"] = "Assigned - INELIGIBLE"
            else:
                entry["status_label"] = "Assigned - Eligible"
            assigned_crew_details.append(entry)
        elif result.eligible:
            if is_busy_elsewhere:
                entry["status_label"] = "Eligible - Assigned Elsewhere"
                eligible_assigned_elsewhere.append(entry)
            else:
                entry["status_label"] = "Eligible - Standby"
                eligible_standby.append(entry)
        else:
            if is_busy_elsewhere:
                entry["status_label"] = "Ineligible - Assigned Elsewhere"
            else:
                entry["status_label"] = "Ineligible - Standby"
            ineligible.append(entry)

    eligible_standby.sort(key=lambda x: x["cost"])

    return {
        "status": "success",
        "flight_id": flight_id,
        "flight_info": {
            "origin": flight.origin,
            "destination": flight.destination,
            "aircraft_type": flight.aircraft_type,
            "flight_hours": flight.flight_hours,
            "is_night_duty": flight.is_night_duty,
            "std": flight.std.strftime("%H:%M"),
        },
        "assigned_crew": assigned_crew_details,
        "eligible_standby": eligible_standby,
        "eligible_assigned_elsewhere": eligible_assigned_elsewhere,
        "ineligible": ineligible,
        "summary": {
            "total_crew": len(all_crew),
            "assigned_to_flight": len(assigned_crew_details),
            "eligible_standby_count": len(eligible_standby),
            "eligible_assigned_elsewhere_count": len(eligible_assigned_elsewhere),
            "ineligible_count": len(ineligible),
        },
    }


def _build_rule_basis(delay_minutes: int) -> str:
    try:
        from rag_engine import retrieve_legal_guidance
    except Exception:
        return (
            "Replacement crew are ranked using the roster's legal-rest status, duty-hour limits, "
            "rolling 7-day hour limits, night-duty constraints, and aircraft-type rating checks."
        )

    query = (
        "What DGCA duty, rest, and compliance rules should be applied when a flight is delayed and "
        "replacement crew must be selected from standby staff?"
    )
    try:
        return retrieve_legal_guidance(query)[:900]
    except Exception:
        return (
            "Replacement crew are ranked using the roster's legal-rest status, duty-hour limits, "
            "rolling 7-day hour limits, night-duty constraints, and aircraft-type rating checks."
        )


def process_delay_with_replacements(
    flight_id: str,
    delay_minutes: int,
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from data.staff_manager import REQUIRED_CREW

    delay_result = process_delay(flight_id, delay_minutes, csv_path, db_path)
    if delay_result.get("status") != "success":
        return delay_result

    replacement_result = find_replacement_crew(flight_id, csv_path, db_path)
    replacement_assignments: List[Dict[str, Any]] = []
    standby_flags: List[Dict[str, Any]] = []
    assignment_errors: List[str] = []

    if replacement_result.get("status") == "success":
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

        if missing:
            for c in replacement_result.get("eligible_standby", []):
                role_name = c.get("role")
                if role_name in missing and missing[role_name] > 0:
                    assign_result = assign_crew_to_flight(c["crew_id"], flight_id, role_name, db_path)
                    if assign_result.get("status") == "success":
                        replacement_assignments.append({
                            **c,
                            "assigned_role": role_name,
                        })
                        missing[role_name] -= 1
                    else:
                        c["assignment_error"] = assign_result.get("message", "Assignment failed")
                        standby_flags.append(c)
                        assignment_errors.append(c["assignment_error"])
                elif role_name not in missing:
                    standby_flags.append(c)
            for c in replacement_result.get("ineligible", []):
                standby_flags.append(c)

    rule_basis = _build_rule_basis(delay_minutes)

    return {
        **delay_result,
        "replacement_plan": replacement_result,
        "replacement_assignments": replacement_assignments,
        "standby_flags": standby_flags,
        "assignment_errors": assignment_errors,
        "rule_basis": rule_basis,
    }


def proactive_crew_assignment(
    today_schedule: List[Dict[str, Any]],
    csv_path: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from data.staff_manager import REQUIRED_CREW

    all_crew = load_crew(csv_path)

    flights_needing_coverage = []
    standby_alerts = []
    crew_recommendations = {}

    for flight in today_schedule:
        pred = flight.get("prediction", {})
        risk = pred.get("risk_level", "Low")
        fid = flight.get("callsign", "")
        exp_delay = pred.get("expected_delay_min", 0)

        if risk not in ("High", "Medium"):
            continue

        flight_hours = flight.get("avg_duration_min", 120) / 60.0
        dep_hour = flight.get("avg_departure_hour", 12)
        is_night = dep_hour < 6 or dep_hour >= 22

        eligible_standby = []
        for member in all_crew:
            if member.rest_status.lower() != "legal":
                continue
            if member.current_duty_hours >= 11.5:
                continue
            result = check_crew_eligibility(
                member,
                scenario_flight_hours=flight_hours,
                scenario_is_night_duty=is_night,
            )
            if result.eligible:
                cost = round(compute_cost(member, flight_hours), 2)
                eligible_standby.append({
                    "crew_id": member.crew_id,
                    "name": member.name,
                    "role": member.role.value,
                    "cost": cost,
                    "rest_status": member.rest_status,
                    "current_duty_hours": member.current_duty_hours,
                    "rolling_7_day_hours": member.rolling_7_day_hours,
                })

        eligible_standby.sort(key=lambda x: x["cost"])

        suggestions = {}
        for role_name in REQUIRED_CREW:
            candidates = [c for c in eligible_standby if c.get("role") == role_name]
            if candidates:
                best = candidates[0]
                suggestions[role_name] = {
                    "crew_id": best["crew_id"],
                    "name": best["name"],
                    "cost": best["cost"],
                    "rest_status": best["rest_status"],
                    "duty_hours": best["current_duty_hours"],
                    "rolling_7d": best["rolling_7_day_hours"],
                }

        rec = {
            "flight_id": fid,
            "route": flight.get("route", ""),
            "scheduled_departure": flight.get("scheduled_departure", ""),
            "risk_level": risk,
            "delay_probability": pred.get("delay_probability", 0),
            "expected_delay_min": exp_delay,
            "factors": pred.get("factors", []),
            "suggested_crew": suggestions,
            "standby_count": len(eligible_standby),
        }

        crew_recommendations[fid] = rec

        if risk == "High":
            flights_needing_coverage.append(rec)
        else:
            standby_alerts.append(rec)

    return {
        "flights_needing_coverage": flights_needing_coverage,
        "standby_alerts": standby_alerts,
        "crew_recommendations": crew_recommendations,
        "summary": {
            "high_risk_count": len(flights_needing_coverage),
            "medium_risk_count": len(standby_alerts),
            "low_risk_count": len(today_schedule) - len(flights_needing_coverage) - len(standby_alerts),
        },
    }
