from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pulp


MAX_DUTY_HOURS = 12.0
MAX_ROLLING_7_DAY_HOURS = 35.0
MAX_CONSECUTIVE_NIGHT_SHIFTS = 2
DEFAULT_SCENARIO_FLIGHT_HOURS = 2.0
DEFAULT_REQUIRED_COUNTS = {
    "Captain": 1,
    "FO": 1,
    "CabinCrew": 2,
    "GroundStaff": 1,
}

ROLE_ALIASES = {
    "CAPTAIN": "Captain",
    "CPT": "Captain",
    "FO": "FO",
    "FIRST OFFICER": "FO",
    "FIRST_OFFICER": "FO",
    "FIRSTOFFICER": "FO",
    "CABIN CREW": "CabinCrew",
    "CABINCREW": "CabinCrew",
    "FLIGHT ATTENDANT": "CabinCrew",
    "GROUND STAFF": "GroundStaff",
    "GROUNDSTAFF": "GroundStaff",
}


@dataclass(frozen=True)
class CrewMember:
    crew_id: str
    name: str
    role: str
    current_duty_hours: float
    rolling_7_day_hours: float
    consecutive_night_shifts: int
    rest_status: str
    cost_multiplier: float

    @property
    def projected_duty_hours(self) -> float:
        return self.current_duty_hours


def normalize_role(value: str) -> str:
    cleaned = " ".join(str(value or "").strip().upper().split())
    if not cleaned:
        return "Unknown"
    return ROLE_ALIASES.get(cleaned, cleaned.title().replace(" ", ""))


def normalize_rest_status(value: str) -> str:
    return "Legal" if str(value or "").strip().lower() == "legal" else "Illegal"


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_crew(csv_path: str | Path) -> List[CrewMember]:
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        crew: List[CrewMember] = []
        for row in reader:
            crew_id = str(row.get("Crew_ID") or row.get("Pilot_ID") or "").strip()
            name = str(row.get("Name") or "").strip()
            role_source = row.get("Role") or row.get("Rank") or "Unknown"
            crew.append(
                CrewMember(
                    crew_id=crew_id,
                    name=name,
                    role=normalize_role(role_source),
                    current_duty_hours=parse_float(row.get("Current_Duty_Hours")),
                    rolling_7_day_hours=parse_float(row.get("Rolling_7_Day_Hours")),
                    consecutive_night_shifts=int(parse_float(row.get("Consecutive_Night_Shifts"))),
                    rest_status=normalize_rest_status(row.get("Rest_Status")),
                    cost_multiplier=parse_float(row.get("Cost_Multiplier"), default=1.0),
                )
            )
    return crew


def eligibility_reason(member: CrewMember, scenario_flight_hours: float, scenario_is_night_duty: bool) -> Optional[str]:
    if member.rest_status != "Legal":
        return "Rest status is illegal"
    if member.current_duty_hours + scenario_flight_hours > MAX_DUTY_HOURS:
        return f"Projected duty hours exceed {MAX_DUTY_HOURS}"
    if member.rolling_7_day_hours + scenario_flight_hours > MAX_ROLLING_7_DAY_HOURS:
        return f"Rolling 7-day hours exceed {MAX_ROLLING_7_DAY_HOURS}"
    if scenario_is_night_duty and member.consecutive_night_shifts >= MAX_CONSECUTIVE_NIGHT_SHIFTS:
        return f"Consecutive night duties would exceed {MAX_CONSECUTIVE_NIGHT_SHIFTS}"
    return None


def required_counts_for_roles(required_counts: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    merged = dict(DEFAULT_REQUIRED_COUNTS)
    if required_counts:
        for role, count in required_counts.items():
            merged[normalize_role(role)] = int(count)
    return merged


def solve_assignment(
    crew_members: Iterable[CrewMember],
    scenario_flight_hours: float = DEFAULT_SCENARIO_FLIGHT_HOURS,
    scenario_is_night_duty: bool = False,
    required_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    crew = list(crew_members)
    required = required_counts_for_roles(required_counts)

    status_map: Dict[str, str] = {}
    eligible_indices: List[int] = []

    for index, member in enumerate(crew):
        reason = eligibility_reason(member, scenario_flight_hours, scenario_is_night_duty)
        if reason is None:
            eligible_indices.append(index)
            status_map[member.crew_id] = "Eligible"
        else:
            status_map[member.crew_id] = reason

    model = pulp.LpProblem("airline_standby_assignment", pulp.LpMinimize)
    decision_vars = {
        index: pulp.LpVariable(f"assign_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in range(len(crew))
    }
    rolling_violation_vars = {
        index: pulp.LpVariable(f"rolling_violation_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in range(len(crew))
    }
    night_violation_vars = {
        index: pulp.LpVariable(f"night_violation_{index}", lowBound=0, upBound=1, cat="Binary")
        for index in range(len(crew))
    }

    model += pulp.lpSum(
        crew[index].cost_multiplier * decision_vars[index]
        + 1000 * rolling_violation_vars[index]
        + 1000 * night_violation_vars[index]
        for index in range(len(crew))
    )

    for index, member in enumerate(crew):
        if index not in eligible_indices:
            model += decision_vars[index] == 0
            model += rolling_violation_vars[index] == 0
            model += night_violation_vars[index] == 0
        else:
            model += rolling_violation_vars[index] == 0
            model += night_violation_vars[index] == 0

    for role, minimum_required in required.items():
        model += (
            pulp.lpSum(
                decision_vars[index]
                for index, member in enumerate(crew)
                if member.role == role and index in eligible_indices
            )
            >= minimum_required,
            f"min_{role.lower()}_required",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    selected: List[Dict[str, Any]] = []
    selected_by_role: Dict[str, int] = {role: 0 for role in required}

    for index, member in enumerate(crew):
        if pulp.value(decision_vars[index]) == 1:
            selected.append(
                {
                    **asdict(member),
                    "projected_duty_hours": round(member.current_duty_hours + scenario_flight_hours, 2),
                    "projected_rolling_7_day_hours": round(member.rolling_7_day_hours + scenario_flight_hours, 2),
                    "projected_consecutive_night_shifts": member.consecutive_night_shifts + (1 if scenario_is_night_duty else 0),
                }
            )
            selected_by_role[member.role] = selected_by_role.get(member.role, 0) + 1

    missing_roles = {
        role: max(required_count - selected_by_role.get(role, 0), 0)
        for role, required_count in required.items()
    }

    status = pulp.LpStatus.get(model.status, "Unknown")
    result = {
        "status": status,
        "scenario_flight_hours": scenario_flight_hours,
        "scenario_is_night_duty": scenario_is_night_duty,
        "max_duty_hours": MAX_DUTY_HOURS,
        "max_rolling_7_day_hours": MAX_ROLLING_7_DAY_HOURS,
        "max_consecutive_night_shifts": MAX_CONSECUTIVE_NIGHT_SHIFTS,
        "required_counts": required,
        "objective_value": round(float(pulp.value(model.objective) or 0.0), 2),
        "selected_count": len(selected),
        "selected_crew": selected,
        "selected_by_role": selected_by_role,
        "missing_roles": missing_roles,
        "crew_status": status_map,
    }

    if status != "Optimal":
        result["message"] = (
            "No optimal assignment was found with the current crew pool. "
            "Check legal rest, duty-hour caps, and role coverage."
        )

    return result


def solve_from_csv(
    csv_path: str | Path,
    scenario_flight_hours: float = DEFAULT_SCENARIO_FLIGHT_HOURS,
    scenario_is_night_duty: bool = False,
    required_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    return solve_assignment(
        load_crew(csv_path),
        scenario_flight_hours=scenario_flight_hours,
        scenario_is_night_duty=scenario_is_night_duty,
        required_counts=required_counts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve the airline standby assignment problem.")
    parser.add_argument("--csv", default="crew_standby_list.csv", help="Path to the crew CSV file")
    parser.add_argument(
        "--scenario-flight-hours",
        type=float,
        default=DEFAULT_SCENARIO_FLIGHT_HOURS,
        help="Hours required for the disruption flight scenario",
    )
    parser.add_argument(
        "--night-duty",
        action="store_true",
        help="Treat the scenario as a night duty for consecutive-night validation",
    )
    parser.add_argument("--captains", type=int, default=DEFAULT_REQUIRED_COUNTS["Captain"])
    parser.add_argument("--fos", type=int, default=DEFAULT_REQUIRED_COUNTS["FO"])
    parser.add_argument("--cabin-crew", type=int, default=DEFAULT_REQUIRED_COUNTS["CabinCrew"])
    parser.add_argument("--ground-staff", type=int, default=DEFAULT_REQUIRED_COUNTS["GroundStaff"])
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    required = {
        "Captain": args.captains,
        "FO": args.fos,
        "CabinCrew": args.cabin_crew,
        "GroundStaff": args.ground_staff,
    }
    result = solve_from_csv(
        args.csv,
        scenario_flight_hours=args.scenario_flight_hours,
        scenario_is_night_duty=args.night_duty,
        required_counts=required,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()