from __future__ import annotations

from data.models import CrewMember, Qualification, Role


def _mk(cid: str, role: Role, quals=(), duty: float = 1.0, rolling: float = 5.0) -> CrewMember:
    return CrewMember(
        crew_id=cid,
        name=cid,
        role=role,
        current_duty_hours=duty,
        rolling_7_day_hours=rolling,
        consecutive_night_shifts=0,
        rest_status="Legal",
        base_cost=100.0,
        overtime_multiplier=1.5,
        qualifications=[Qualification(q) for q in quals],
        base_airport="BLR",
        seniority=1,
        hours_flown_30_days=50.0,
        days_since_rest=3,
        consecutive_days_on=0,
    )


def _squad():
    crew = []
    for i in range(3):
        crew.append(_mk(f"CAP{i}", Role.CAPTAIN, quals=("B737",)))
        crew.append(_mk(f"FO{i}", Role.FO, quals=("B737",)))
        crew.append(_mk(f"CC{i}", Role.CABIN_CREW))
        crew.append(_mk(f"RA{i}", Role.RAMP_AGENT))
        crew.append(_mk(f"BH{i}", Role.BAGGAGE_HANDLER))
        crew.append(_mk(f"CL{i}", Role.CABIN_CLEANER))
        crew.append(_mk(f"CI{i}", Role.CHECKIN_AGENT))
        crew.append(_mk(f"SA{i}", Role.SECURITY_AGENT))
    return crew


def test_solve_assignment_fills_roles():
    from solver import solve_assignment

    result = solve_assignment(_squad(), scenario_flight_hours=2.0)
    assert result["status"] == "Optimal"
    assert result["selected_by_role"]["Captain"] >= 1
    assert result["selected_by_role"]["FO"] >= 1
    assert result["selected_by_role"]["CabinCrew"] >= 2
    assert all(n == 0 for n in result["missing_roles"].values())


def test_solve_assignment_crew_status_keys():
    from solver import solve_assignment

    result = solve_assignment(_squad(), scenario_flight_hours=2.0)
    assert set(result["crew_status"].keys()) == {c.crew_id for c in _squad()}


def test_solve_from_csv(crew_csv):
    from solver import solve_from_csv

    result = solve_from_csv(str(crew_csv))
    assert "selected_crew" in result
    assert result["selected_count"] >= 0
