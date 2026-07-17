from __future__ import annotations

import csv
import json
import pulp
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data.models import CrewMember, Flight, FlightStatus, Role, Qualification
from data.crew_loader import load_crew
from data.flights_db import get_flights, get_flight
from validators.dgca_validator import (
    MAX_DUTY_HOURS,
    MAX_ROLLING_7_DAY_HOURS,
    MAX_CONSECUTIVE_NIGHT_SHIFTS,
    STANDARD_DUTY_THRESHOLD,
    MIN_REST_HOURS,
    check_crew_eligibility,
    compute_cost,
    ComplianceResult,
)

DEFAULT_CSV_PATH = Path(__file__).parent / "crew_standby_list.csv"
DEFAULT_SCENARIO_FLIGHT_HOURS = 2.0
DEFAULT_SCENARIO_IS_NIGHT = False
DEFAULT_REQUIRED_COUNTS = {
    Role.CAPTAIN.value: 1,
    Role.FO.value: 1,
    Role.CABIN_CREW.value: 2,
    Role.GROUND_STAFF.value: 1,
}


def solve_from_csv(
    csv_path: str | Path,
    scenario_flight_hours: float = DEFAULT_SCENARIO_FLIGHT_HOURS,
    scenario_is_night_duty: bool = DEFAULT_SCENARIO_IS_NIGHT,
    required_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    crew = load_crew(csv_path)
    return solve_assignment(
        crew,
        scenario_flight_hours=scenario_flight_hours,
        scenario_is_night_duty=scenario_is_night_duty,
        required_counts=required_counts,
    )


def solve_assignment(
    crew_members: Iterable[CrewMember],
    scenario_flight_hours: float = DEFAULT_SCENARIO_FLIGHT_HOURS,
    scenario_is_night_duty: bool = DEFAULT_SCENARIO_IS_NIGHT,
    required_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    crew = list(crew_members)
    required = dict(required_counts or DEFAULT_REQUIRED_COUNTS)

    crew_status: Dict[str, str] = {}
    eligible_indices: List[int] = []
    for idx, member in enumerate(crew):
        result = check_crew_eligibility(
            member,
            scenario_flight_hours=scenario_flight_hours,
            scenario_is_night_duty=scenario_is_night_duty,
        )
        if result.eligible:
            eligible_indices.append(idx)
            crew_status[member.crew_id] = "Eligible"
        else:
            crew_status[member.crew_id] = "; ".join(result.violations)

    model = pulp.LpProblem("airline_standby_assignment", pulp.LpMinimize)
    assign = {
        i: pulp.LpVariable(f"assign_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(len(crew))
    }

    model += pulp.lpSum(
        compute_cost(crew[i], scenario_flight_hours) * assign[i]
        for i in range(len(crew))
    )

    for i in range(len(crew)):
        if i not in eligible_indices:
            model += assign[i] == 0

    for role_name, min_count in required.items():
        role = Role(role_name) if role_name in [r.value for r in Role] else Role.CAPTAIN
        model += (
            pulp.lpSum(
                assign[i]
                for i in eligible_indices
                if crew[i].role == role
            )
            >= min_count,
            f"min_{role_name.lower()}_required",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    selected: List[Dict[str, Any]] = []
    selected_by_role: Dict[str, int] = {r: 0 for r in required}

    for i in range(len(crew)):
        if pulp.value(assign[i]) == 1:
            m = crew[i]
            entry = {
                **asdict(m),
                "Role": m.role.value,
                "qualifications": [q.to_dict() for q in m.qualifications],
                "Projected_Duty_Hours": round(m.current_duty_hours + scenario_flight_hours, 2),
                "Projected_Rolling_7_Day_Hours": round(m.rolling_7_day_hours + scenario_flight_hours, 2),
                "Projected_Consecutive_Night_Shifts": m.consecutive_night_shifts + (1 if scenario_is_night_duty else 0),
                "Selection_Cost": round(compute_cost(m, scenario_flight_hours), 2),
            }
            selected.append(entry)
            selected_by_role[m.role.value] = selected_by_role.get(m.role.value, 0) + 1

    missing_roles = {
        r: max(c - selected_by_role.get(r, 0), 0)
        for r, c in required.items()
    }

    return {
        "status": pulp.LpStatus.get(model.status, "Unknown"),
        "scenario_flight_hours": scenario_flight_hours,
        "scenario_is_night_duty": scenario_is_night_duty,
        "required_counts": required,
        "objective_value": round(float(pulp.value(model.objective) or 0.0), 2),
        "selected_count": len(selected),
        "selected_crew": selected,
        "selected_by_role": selected_by_role,
        "missing_roles": missing_roles,
        "crew_status": crew_status,
    }


def solve_multi_flight(
    flight_ids: List[str],
    csv_path: str | Path = DEFAULT_CSV_PATH,
    required_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    crew = load_crew(csv_path)
    flights = [get_flight(fid) for fid in flight_ids]
    flights = [f for f in flights if f is not None]

    if not flights:
        return {"status": "Error", "message": "No valid flights found", "selected_crew": []}

    required = dict(required_counts or DEFAULT_REQUIRED_COUNTS)

    crew_status: Dict[str, str] = {}
    eligible_indices: List[int] = []
    for idx, member in enumerate(crew):
        total_hours = sum(f.flight_hours for f in flights)
        is_night = any(f.is_night_duty for f in flights)
        result = check_crew_eligibility(
            member, flights=flights,
            scenario_flight_hours=total_hours,
            scenario_is_night_duty=is_night,
        )
        if result.eligible:
            eligible_indices.append(idx)
            crew_status[member.crew_id] = "Eligible"
        else:
            crew_status[member.crew_id] = "; ".join(result.violations)

    model = pulp.LpProblem("airline_multi_flight_assignment", pulp.LpMinimize)
    assign = {
        i: pulp.LpVariable(f"assign_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(len(crew))
    }

    total_flight_hours = sum(f.flight_hours for f in flights)
    model += pulp.lpSum(
        compute_cost(crew[i], total_flight_hours) * assign[i]
        for i in range(len(crew))
    )

    for i in range(len(crew)):
        if i not in eligible_indices:
            model += assign[i] == 0

    for role_name, min_count in required.items():
        try:
            role = Role(role_name)
        except ValueError:
            continue
        model += (
            pulp.lpSum(
                assign[i]
                for i in eligible_indices
                if crew[i].role == role
            )
            >= min_count,
            f"min_{role_name.lower()}_required",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    selected: List[Dict[str, Any]] = []
    selected_by_role: Dict[str, int] = {r: 0 for r in required}

    for i in range(len(crew)):
        if pulp.value(assign[i]) == 1:
            m = crew[i]
            entry = {
                **asdict(m),
                "Role": m.role.value,
                "qualifications": [q.to_dict() for q in m.qualifications],
                "Total_Flight_Hours": round(total_flight_hours, 2),
                "Projected_Duty_Hours": round(m.current_duty_hours + total_flight_hours, 2),
                "Projected_Rolling_7_Day_Hours": round(m.rolling_7_day_hours + total_flight_hours, 2),
                "Selection_Cost": round(compute_cost(m, total_flight_hours), 2),
            }
            selected.append(entry)
            selected_by_role[m.role.value] = selected_by_role.get(m.role.value, 0) + 1

    missing_roles = {
        r: max(c - selected_by_role.get(r, 0), 0)
        for r, c in required.items()
    }

    return {
        "status": pulp.LpStatus.get(model.status, "Unknown"),
        "flight_ids": flight_ids,
        "flight_count": len(flights),
        "total_flight_hours": round(total_flight_hours, 2),
        "is_night_duty": any(f.is_night_duty for f in flights),
        "required_counts": required,
        "objective_value": round(float(pulp.value(model.objective) or 0.0), 2),
        "selected_count": len(selected),
        "selected_crew": selected,
        "selected_by_role": selected_by_role,
        "missing_roles": missing_roles,
        "crew_status": crew_status,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Airline crew assignment solver")
    parser.add_argument("--csv", default="crew_standby_list.csv")
    parser.add_argument("--scenario-flight-hours", type=float, default=DEFAULT_SCENARIO_FLIGHT_HOURS)
    parser.add_argument("--night-duty", action="store_true")
    parser.add_argument("--captains", type=int, default=DEFAULT_REQUIRED_COUNTS[Role.CAPTAIN.value])
    parser.add_argument("--fos", type=int, default=DEFAULT_REQUIRED_COUNTS[Role.FO.value])
    parser.add_argument("--cabin-crew", type=int, default=DEFAULT_REQUIRED_COUNTS[Role.CABIN_CREW.value])
    parser.add_argument("--ground-staff", type=int, default=DEFAULT_REQUIRED_COUNTS[Role.GROUND_STAFF.value])
    parser.add_argument("--multi-flight", nargs="+", help="Flight IDs for multi-flight assignment")
    args = parser.parse_args()

    required = {
        Role.CAPTAIN.value: args.captains,
        Role.FO.value: args.fos,
        Role.CABIN_CREW.value: args.cabin_crew,
        Role.GROUND_STAFF.value: args.ground_staff,
    }

    if args.multi_flight:
        result = solve_multi_flight(args.multi_flight, args.csv, required)
    else:
        result = solve_from_csv(
            args.csv,
            scenario_flight_hours=args.scenario_flight_hours,
            scenario_is_night_duty=args.night_duty,
            required_counts=required,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
