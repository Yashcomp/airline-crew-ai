from __future__ import annotations

from typing import Any, Dict, List, Optional

from data.models import CrewMember, Flight, Role
from data.crew_loader import load_crew
from data.flights_db import get_flights, get_flight
from validators.dgca_validator import check_crew_eligibility, compute_cost
from pathlib import Path

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "crew_standby_list.csv"


def find_eligible_crew_for_flight(
    flight_id: str,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    flight = get_flight(flight_id)
    if not flight:
        return {"error": f"Flight {flight_id} not found"}

    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    eligible = []
    ineligible = []

    for member in crew:
        result = check_crew_eligibility(
            member,
            flights=[flight],
            scenario_flight_hours=flight.flight_hours,
            scenario_is_night_duty=flight.is_night_duty,
        )
        entry = {
            **member.to_dict(),
            "cost": round(compute_cost(member, flight.flight_hours), 2),
        }
        if result.eligible:
            eligible.append(entry)
        else:
            entry["violations"] = result.violations
            ineligible.append(entry)

    eligible.sort(key=lambda x: x["cost"])

    return {
        "flight": flight.to_dict(),
        "eligible_count": len(eligible),
        "ineligible_count": len(ineligible),
        "eligible_crew": eligible,
        "ineligible_crew": ineligible,
    }


def answer_crew_for_flight_query(question: str, csv_path: Optional[str] = None) -> str:
    import re
    match = re.search(r"([A-Z]{2,3}[-_]?\d{2,4})", question, re.IGNORECASE)
    if not match:
        return "Please specify a flight ID (e.g., 'AI-302') to check crew availability."

    flight_id = match.group(1).upper().replace("_", "-")
    result = find_eligible_crew_for_flight(flight_id, csv_path)
    if "error" in result:
        return result["error"]

    flight = result["flight"]
    lines = [f"**Flight {flight['flight_id']}** ({flight['origin']}->{flight['destination']}) - {flight['aircraft_type']}"]
    lines.append(f"Duration: {flight['flight_hours']}h | Night duty: {'Yes' if flight['is_night_duty'] else 'No'}")
    lines.append("")

    if result["eligible_crew"]:
        lines.append(f"**{result['eligible_count']} eligible crew:**")
        for c in result["eligible_crew"][:10]:
            lines.append(f"  {c['name']} ({c['role']}) - Cost: ${c['cost']:.2f}")
    else:
        lines.append("**No eligible crew found for this flight.**")

    if result["ineligible_crew"]:
        lines.append(f"\n**{result['ineligible_count']} ineligible:**")
        for c in result["ineligible_crew"][:5]:
            reasons = "; ".join(c.get("violations", []))
            lines.append(f"  {c['name']} ({c['role']}): {reasons}")

    return "\n".join(lines)
