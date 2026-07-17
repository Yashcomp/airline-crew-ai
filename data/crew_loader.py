from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.models import CrewMember, Qualification, Role

ROLE_ALIASES = {
    "CAPTAIN": Role.CAPTAIN,
    "CPT": Role.CAPTAIN,
    "FO": Role.FO,
    "FIRST OFFICER": Role.FO,
    "FIRST_OFFICER": Role.FO,
    "FIRSTOFFICER": Role.FO,
    "CABIN CREW": Role.CABIN_CREW,
    "CABINCREW": Role.CABIN_CREW,
    "FLIGHT ATTENDANT": Role.CABIN_CREW,
    "GROUND STAFF": Role.GROUND_STAFF,
    "GROUNDSTAFF": Role.GROUND_STAFF,
}


def normalize_role(value: Any) -> Role:
    cleaned = " ".join(str(value or "").strip().upper().split())
    role = ROLE_ALIASES.get(cleaned)
    if role:
        return role
    try:
        return Role(cleaned.title().replace(" ", ""))
    except ValueError:
        return Role.CABIN_CREW


def _parse_float(value: Any, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value: Any, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_qualifications(raw: Optional[str]) -> List[Qualification]:
    if not raw or str(raw).strip() == "":
        return []
    quals = []
    for token in str(raw).split(";"):
        token = token.strip()
        if token:
            quals.append(Qualification(aircraft_type=token))
    return quals


def load_crew(csv_path: str | Path) -> List[CrewMember]:
    path = Path(csv_path)
    crew: List[CrewMember] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(h.strip() for h in (reader.fieldnames or []))
        is_new_schema = "qualifications" in headers or "base_airport" in headers

        for row in reader:
            row = {k.strip(): v for k, v in row.items()} if row else {}

            if is_new_schema:
                member = CrewMember(
                    crew_id=str(row.get("crew_id", row.get("Crew_ID", ""))).strip(),
                    name=str(row.get("name", row.get("Name", ""))).strip(),
                    role=normalize_role(row.get("role", row.get("Rank", ""))),
                    current_duty_hours=_parse_float(
                        row.get("current_duty_hours", row.get("Current_Duty_Hours", 0))
                    ),
                    rolling_7_day_hours=_parse_float(
                        row.get("rolling_7_day_hours", row.get("Rolling_7_Day_Hours", 0))
                    ),
                    consecutive_night_shifts=_parse_int(
                        row.get("consecutive_night_shifts", row.get("Consecutive_Night_Shifts", 0))
                    ),
                    rest_status=str(row.get("rest_status", row.get("Rest_Status", "Legal"))).strip().capitalize(),
                    base_cost=_parse_float(
                        row.get("base_cost", row.get("Base_Cost", 1.0)), 1.0
                    ),
                    overtime_multiplier=_parse_float(
                        row.get("overtime_multiplier", row.get("Overtime_Multiplier", 1.0)), 1.0
                    ),
                    qualifications=_parse_qualifications(
                        row.get("qualifications", row.get("Qualifications", ""))
                    ),
                    base_airport=str(row.get("base_airport", row.get("Base_Airport", ""))).strip().upper(),
                    seniority=_parse_int(row.get("seniority", row.get("Seniority", 0))),
                    hours_flown_30_days=_parse_float(
                        row.get("hours_flown_30_days", row.get("Hours_Flown_30_Days", 0))
                    ),
                    days_since_rest=_parse_int(
                        row.get("days_since_rest", row.get("Days_Since_Rest", 0))
                    ),
                    consecutive_days_on=_parse_int(
                        row.get("consecutive_days_on", row.get("Consecutive_Days_On", 0))
                    ),
                )
            else:
                member = CrewMember(
                    crew_id=str(row.get("Crew_ID", "")).strip(),
                    name=str(row.get("Name", "")).strip(),
                    role=normalize_role(row.get("Rank", row.get("Role", ""))),
                    current_duty_hours=_parse_float(row.get("Current_Duty_Hours", 0)),
                    rolling_7_day_hours=_parse_float(row.get("Rolling_7_Day_Hours", 0)),
                    consecutive_night_shifts=_parse_int(row.get("Consecutive_Night_Shifts", 0)),
                    rest_status=str(row.get("Rest_Status", "Legal")).strip().capitalize(),
                    base_cost=_parse_float(row.get("Base_Cost", 1.0), 1.0),
                    overtime_multiplier=_parse_float(row.get("Overtime_Multiplier", 1.0), 1.0),
                )

            if member.crew_id and member.name:
                crew.append(member)

    return crew


def crew_to_dicts(crew: List[CrewMember]) -> List[Dict[str, Any]]:
    return [m.to_dict() for m in crew]
