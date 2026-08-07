from __future__ import annotations

from data.models import Role
from validators.dgca_validator import (
    check_crew_eligibility,
    check_cumulative_limits,
    check_weekly_rest,
    compute_cost,
    scenario_hours_for_role,
)


def test_scenario_hours_airborne_role():
    assert scenario_hours_for_role(Role.CABIN_CREW, 2.0) == 2.0


def test_scenario_hours_ground_role():
    assert scenario_hours_for_role(Role.RAMP_AGENT, 2.0) == 0.0


def test_eligible_crew(captain):
    result = check_crew_eligibility(captain, scenario_flight_hours=1.0)
    assert result.eligible
    assert not result.violations


def test_rest_status_violation(captain):
    captain.rest_status = "Fatigued"
    result = check_crew_eligibility(captain, scenario_flight_hours=1.0)
    assert not result.eligible
    assert any("Rest status" in v for v in result.violations)


def test_duty_limit_violation(captain):
    result = check_crew_eligibility(captain, scenario_flight_hours=99.0)
    assert not result.eligible
    assert any("duty hours" in v.lower() for v in result.violations)


def test_rolling_limit_violation(captain):
    captain.rolling_7_day_hours = 34.0
    result = check_crew_eligibility(captain, scenario_flight_hours=3.0)
    assert not result.eligible
    assert any("Rolling 7-day" in v for v in result.violations)


def test_compute_cost(captain):
    assert compute_cost(captain, 2.0) >= 0


def test_cumulative_limits_result_shape(captain):
    result = check_cumulative_limits(captain)
    assert hasattr(result, "eligible")
    assert isinstance(result.violations, list)


def test_weekly_rest_result_shape(captain):
    result = check_weekly_rest(captain)
    assert hasattr(result, "eligible")
