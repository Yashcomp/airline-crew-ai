from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.models import CrewMember, Flight, FlightStatus
from data.flights_db import get_flights, get_flight, get_disrupted_flights, get_upcoming_flights, get_flight_stats
from data.crew_loader import load_crew
from pathlib import Path


DEFAULT_CSV_PATH = Path(__file__).parent.parent / "crew_standby_list.csv"


def query_flights(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
    status: Optional[str] = None,
    aircraft_type: Optional[str] = None,
) -> Dict[str, Any]:
    from datetime import datetime as dt
    parsed_date = dt.fromisoformat(date) if date else None
    flights = get_flights(
        origin=origin,
        destination=destination,
        date=parsed_date,
        status=status,
        aircraft_type=aircraft_type,
    )
    return {
        "count": len(flights),
        "flights": [f.to_dict() for f in flights],
    }


def get_disruption_summary() -> Dict[str, Any]:
    disrupted = get_disrupted_flights()
    return {
        "disrupted_count": len(disrupted),
        "flights": [f.to_dict() for f in disrupted],
    }


def get_crew_availability(
    csv_path: Optional[str] = None,
    flight: Optional[Flight] = None,
) -> Dict[str, Any]:
    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    result = []
    for m in crew:
        eligible = True
        reason = None
        if m.rest_status != "Legal":
            eligible = False
            reason = f"Rest status: {m.rest_status}"
        elif flight and not m.is_rated_on(flight.aircraft_type):
            eligible = False
            reason = f"Not rated for {flight.aircraft_type}"

        result.append({
            **m.to_dict(),
            "eligible": eligible,
            "ineligibility_reason": reason,
        })

    eligible_count = sum(1 for r in result if r["eligible"])
    return {
        "total": len(result),
        "eligible": eligible_count,
        "unavailable": len(result) - eligible_count,
        "crew": result,
    }


def answer_crew_query(question: str, csv_path: Optional[str] = None) -> str:
    import re
    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    lowered = question.lower()

    crew_ids = re.findall(r"(CRW\d{3})", question, re.IGNORECASE)
    if crew_ids:
        lines = []
        for cid in crew_ids:
            member = next((m for m in crew if m.crew_id.upper() == cid.upper()), None)
            if member:
                quals = ", ".join(q.aircraft_type for q in member.qualifications) if member.qualifications else "None"
                lines.append(
                    f"**{member.name}** ({member.crew_id})\n"
                    f"  Role: {member.role.value}\n"
                    f"  Base: {member.base_airport or 'N/A'} | Seniority: {member.seniority}\n"
                    f"  Rest Status: {member.rest_status}\n"
                    f"  Current Duty: {member.current_duty_hours}h | Rolling 7-day: {member.rolling_7_day_hours}h/35h\n"
                    f"  Night Shifts: {member.consecutive_night_shifts}/3 | Days Since Rest: {member.days_since_rest}\n"
                    f"  Consecutive Days On: {member.consecutive_days_on}\n"
                    f"  Hours Flown (30 days): {member.hours_flown_30_days}h\n"
                    f"  Qualifications: {quals}\n"
                    f"  Cost: ${member.base_cost:.2f} x {member.overtime_multiplier} OT"
                )
            else:
                lines.append(f"Crew member **{cid.upper()}** not found in the roster.")
        return "\n\n".join(lines)

    name_matches = []
    for m in crew:
        if m.name.lower() in lowered:
            name_matches.append(m)
    if name_matches:
        lines = []
        for member in name_matches:
            quals = ", ".join(q.aircraft_type for q in member.qualifications) if member.qualifications else "None"
            lines.append(
                f"**{member.name}** ({member.crew_id})\n"
                f"  Role: {member.role.value}\n"
                f"  Base: {member.base_airport or 'N/A'} | Seniority: {member.seniority}\n"
                f"  Rest Status: {member.rest_status}\n"
                f"  Current Duty: {member.current_duty_hours}h | Rolling 7-day: {member.rolling_7_day_hours}h/35h\n"
                f"  Night Shifts: {member.consecutive_night_shifts}/3 | Days Since Rest: {member.days_since_rest}\n"
                f"  Qualifications: {quals}"
            )
        return "\n\n".join(lines)

    if "available" in lowered or "free" in lowered or "legal" in lowered:
        available = [m for m in crew if m.rest_status == "Legal"]
        if not available:
            return "No crew members are currently legal for assignment."
        lines = [f"{m.name} ({m.role.value}) - {m.crew_id}: {m.current_duty_hours}h duty, {m.rolling_7_day_hours}h rolling" for m in available]
        return f"**{len(available)} crew members available:**\n" + "\n".join(lines)

    if "captain" in lowered or "cpt" in lowered:
        captains = [m for m in crew if m.role == Role.CAPTAIN]
        if not captains:
            return "No captains in the roster."
        lines = [f"{m.name} ({m.crew_id}): {m.current_duty_hours}h duty, rest={m.rest_status}" for m in captains]
        return f"**{len(captains)} Captain(s):**\n" + "\n".join(lines)

    if "fo" in lowered or "first officer" in lowered:
        fos = [m for m in crew if m.role == Role.FO]
        if not fos:
            return "No first officers in the roster."
        lines = [f"{m.name} ({m.crew_id}): {m.current_duty_hours}h duty, rest={m.rest_status}" for m in fos]
        return f"**{len(fos)} First Officer(s):**\n" + "\n".join(lines)

    if "cabin" in lowered:
        cabin = [m for m in crew if m.role == Role.CABIN_CREW]
        lines = [f"{m.name} ({m.crew_id}): {m.current_duty_hours}h duty, rest={m.rest_status}" for m in cabin]
        return f"**{len(cabin)} Cabin Crew:**\n" + "\n".join(lines)

    if "ground" in lowered:
        ground = [m for m in crew if m.role == Role.GROUND_STAFF]
        lines = [f"{m.name} ({m.crew_id}): {m.current_duty_hours}h duty, rest={m.rest_status}" for m in ground]
        return f"**{len(ground)} Ground Staff:**\n" + "\n".join(lines)

    if "illegal" in lowered or "unavailable" in lowered:
        unavailable = [m for m in crew if m.rest_status != "Legal"]
        if not unavailable:
            return "All crew members are currently legal."
        lines = [f"{m.name} ({m.role.value}, {m.crew_id}): rest={m.rest_status}" for m in unavailable]
        return f"**{len(unavailable)} unavailable crew:**\n" + "\n".join(lines)

    total = len(crew)
    by_role = {}
    for m in crew:
        by_role.setdefault(m.role.value, []).append(m)
    lines = [f"**{role}**: {len(members)} total, {sum(1 for m in members if m.rest_status == 'Legal')} legal" for role, members in by_role.items()]
    return f"**Roster summary ({total} crew):**\n" + "\n".join(lines)


def answer_flight_query(question: str) -> str:
    import re
    lowered = question.lower()
    stats = get_flight_stats()
    disrupted = get_disrupted_flights()
    upcoming = get_upcoming_flights(hours_ahead=12)

    flight_ids = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", question, re.IGNORECASE)
    if flight_ids:
        lines = []
        for fid in flight_ids:
            flight = get_flight(fid.upper().replace("_", "-"))
            if flight is None:
                flight = get_flight(fid.upper())
            if flight:
                lines.append(
                    f"**{flight.flight_id}** ({flight.status.value})\n"
                    f"  Route: {flight.origin} -> {flight.destination}\n"
                    f"  Departure: {flight.std.strftime('%H:%M')} | STA: {flight.sta.strftime('%H:%M')}\n"
                    f"  Aircraft: {flight.aircraft_type} | Gate: {flight.gate or 'N/A'} | Terminal: {flight.terminal or 'N/A'}\n"
                    f"  Passengers: {flight.pax_count} | Duration: {flight.flight_duration_min} min ({flight.flight_hours}h)\n"
                    f"  International: {'Yes' if flight.is_international else 'No'}\n"
                    f"  Night Duty: {'Yes' if flight.is_night_duty else 'No'}"
                    + (f"\n  Disruption: {flight.disruption_reason}" if flight.disruption_reason else "")
                )
            else:
                lines.append(f"Flight **{fid.upper()}** not found in the schedule.")
        return "\n\n".join(lines)

    if "disrupt" in lowered or "delay" in lowered or "cancel" in lowered:
        if not disrupted:
            return "No disrupted flights at this time."
        lines = [f"**{f.flight_id}** {f.origin}->{f.destination} ({f.status.value}): {f.disruption_reason or 'No reason given'}" for f in disrupted]
        return f"**{len(disrupted)} disrupted flight(s):**\n" + "\n".join(lines)

    if "upcoming" in lowered or "next" in lowered:
        if not upcoming:
            return "No upcoming flights in the next 12 hours."
        lines = [f"**{f.flight_id}** {f.origin}->{f.destination} at {f.std.strftime('%H:%M')} ({f.aircraft_type})" for f in upcoming]
        return f"**{len(upcoming)} upcoming flight(s):**\n" + "\n".join(lines)

    total = stats.get("total", 0)
    by_status = stats.get("by_status", {})
    lines = [f"  {status}: {count}" for status, count in by_status.items()]
    return f"**Flight schedule summary ({total} flights):**\n" + "\n".join(lines) if lines else "No flights in the schedule."


from data.models import Role
