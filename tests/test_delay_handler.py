from __future__ import annotations

from datetime import datetime, timezone

from data.delay_handler import process_delay, proactive_crew_assignment
from data.flights_db import insert_flight
from data.models import Flight


def test_process_delay_unknown_flight(db_path):
    result = process_delay("NOPE", 30, "crew.csv", db_path=db_path)
    assert result["status"] == "error"


def test_process_delay_no_crew_assigned(db_path, crew_csv):
    flight = Flight(
        flight_id="FL100",
        origin="BLR",
        destination="DEL",
        std=datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
        aircraft_type="B737",
        flight_duration_min=120,
    )
    insert_flight(flight, db_path=db_path)

    result = process_delay("FL100", 30, str(crew_csv), db_path=db_path)
    assert result["status"] == "success"
    assert "No crew was assigned" in result["message"]
    assert result["unassigned_count"] == 0


def test_proactive_crew_assignment_empty_schedule(crew_csv, db_path):
    result = proactive_crew_assignment([], str(crew_csv), db_path)
    assert "summary" in result
    assert result["summary"]["high_risk_count"] == 0
    assert result["summary"]["low_risk_count"] == 0


def test_proactive_suggests_every_ineligible_member_per_role(tmp_path, db_path):
    from data.flights_db import assign_crew_to_flight, insert_flight
    from data.models import Flight
    from data.staff_manager import REQUIRED_CREW

    csv_path = tmp_path / "crew_standby_list.csv"
    header = (
        "crew_id,name,role,current_duty_hours,rolling_7_day_hours,"
        "consecutive_night_shifts,rest_status,base_cost,overtime_multiplier,"
        "qualifications,base_airport,seniority,hours_flown_30_days,"
        "days_since_rest,consecutive_days_on,shift"
    )
    rows = [
        "CRW001,Arjun Mehta,Captain,0.5,5.0,0,Legal,135.0,1.4,B737;A320,BLR,6,60.0,5,0,",
        "CRW002,Neha Rao,FO,1.0,8.0,0,Legal,95.0,1.2,B737;A320,BLR,4,55.0,4,0,",
        "CRW003,Simran Kaur,CabinCrew,13.5,40.0,0,Legal,70.0,1.1,,BLR,3,30.0,3,0,",
        "CRW004,Rahul Nair,CabinCrew,13.0,40.0,0,Legal,70.0,1.1,,BLR,3,28.0,2,0,",
        "CRW005,Dev Patel,RampAgent,0.0,2.0,0,Legal,45.0,1.0,,BLR,2,0.0,1,0,",
        "CRW006,Anita Das,BaggageHandler,0.0,2.0,0,Legal,45.0,1.0,,BLR,2,0.0,1,0,",
        "CRW007,Suresh Iyer,CabinCleaner,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
        "CRW008,Priya Menon,CheckinAgent,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
        "CRW009,Karan Joshi,SecurityAgent,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
        "CRW010,Mira Chatterjee,CabinCrew,2.0,10.0,0,Legal,70.0,1.1,,BLR,2,5.0,2,0,",
        "CRW011,Aditya Rao,CabinCrew,1.0,9.0,0,Legal,70.0,1.1,,BLR,2,4.0,2,0,",
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    flight = Flight(
        flight_id="FL200",
        origin="BLR",
        destination="DEL",
        std=datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
        aircraft_type="B737",
        flight_duration_min=120,
    )
    insert_flight(flight, db_path=db_path)
    assign_crew_to_flight("CRW003", "FL200", "CabinCrew", db_path=db_path)
    assign_crew_to_flight("CRW004", "FL200", "CabinCrew", db_path=db_path)

    schedule = [{
        "callsign": "FL200",
        "route": "BLR-DEL",
        "avg_duration_min": 120,
        "avg_departure_hour": 12,
        "scheduled_date": "2026-08-07",
        "scheduled_departure": "12:00",
        "scheduled_arrival": "14:00",
        "leg_type": "First Leg",
        "crew_action": "",
        "prediction": {
            "risk_level": "High",
            "expected_delay_min": 60,
            "delay_probability": 0.8,
            "factors": [],
        },
    }]

    result = proactive_crew_assignment(schedule, str(csv_path), db_path)
    rec = result["crew_recommendations"]["FL200"]
    sug = rec["suggested_crew"]

    cc = sug["CabinCrew"]
    assert len(cc) == 2
    assert [c["old_crew_id"] for c in cc] == ["CRW003", "CRW004"]
    assert all(c["old_crew_id"] is not None for c in cc)
    assert len({c["crew_id"] for c in cc}) == len(cc)
    assert all(c["reason"] for c in cc)
    assert not any(c["crew_id"].startswith("STBY-") for c in cc)

    total_suggestions = sum(len(v) for v in sug.values())
    assert total_suggestions == 2
    assert rec["understaffed_roles"]["CabinCrew"] == 2
    assert rec["no_standby_roles"] == []

    reloaded = csv_path.read_text(encoding="utf-8")
    assert "STBY-" not in reloaded


def test_proactive_unstaffed_flight_has_no_replacements(tmp_path, db_path):
    from data.flights_db import insert_flight
    from data.models import Flight
    from data.staff_manager import REQUIRED_CREW

    csv_path = tmp_path / "crew_standby_list.csv"
    header = (
        "crew_id,name,role,current_duty_hours,rolling_7_day_hours,"
        "consecutive_night_shifts,rest_status,base_cost,overtime_multiplier,"
        "qualifications,base_airport,seniority,hours_flown_30_days,"
        "days_since_rest,consecutive_days_on,shift"
    )
    rows = [
        "CRW001,Arjun Mehta,Captain,0.5,5.0,0,Legal,135.0,1.4,B737;A320,BLR,6,60.0,5,0,",
        "CRW002,Neha Rao,FO,1.0,8.0,0,Legal,95.0,1.2,B737;A320,BLR,4,55.0,4,0,",
        "CRW003,Simran Kaur,CabinCrew,0.8,6.0,0,Legal,70.0,1.1,,BLR,3,30.0,3,0,",
        "CRW004,Rahul Nair,CabinCrew,0.2,3.0,0,Legal,70.0,1.1,,BLR,3,28.0,2,0,",
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    flight = Flight(
        flight_id="FL201",
        origin="BLR",
        destination="DEL",
        std=datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
        aircraft_type="B737",
        flight_duration_min=120,
    )
    insert_flight(flight, db_path=db_path)

    schedule = [{
        "callsign": "FL201",
        "route": "BLR-DEL",
        "avg_duration_min": 120,
        "avg_departure_hour": 12,
        "scheduled_date": "2026-08-07",
        "scheduled_departure": "12:00",
        "scheduled_arrival": "14:00",
        "leg_type": "First Leg",
        "crew_action": "",
        "prediction": {
            "risk_level": "High",
            "expected_delay_min": 60,
            "delay_probability": 0.8,
            "factors": [],
        },
    }]

    result = proactive_crew_assignment(schedule, str(csv_path), db_path)
    rec = result["crew_recommendations"]["FL201"]

    assert rec["suggested_crew"] == {}
    assert rec["understaffed_roles"]["CabinCrew"] == 4
    assert rec["understaffed_roles"]["Captain"] == 1
    assert rec["understaffed_roles"]["RampAgent"] == 1
    assert len(rec["understaffed_roles"]) == len(REQUIRED_CREW)
    assert rec["no_standby_roles"] == []


def test_proactive_no_fabricated_standby_when_no_candidates(tmp_path, db_path):
    from data.flights_db import assign_crew_to_flight, insert_flight
    from data.models import Flight

    csv_path = tmp_path / "crew_standby_list.csv"
    header = (
        "crew_id,name,role,current_duty_hours,rolling_7_day_hours,"
        "consecutive_night_shifts,rest_status,base_cost,overtime_multiplier,"
        "qualifications,base_airport,seniority,hours_flown_30_days,"
        "days_since_rest,consecutive_days_on,shift"
    )
    rows = [
        "CRW003,Simran Kaur,CabinCrew,13.5,40.0,0,Legal,70.0,1.1,,BLR,3,30.0,3,0,",
    ]
    csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    flight = Flight(
        flight_id="FL202",
        origin="BLR",
        destination="DEL",
        std=datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
        aircraft_type="B737",
        flight_duration_min=120,
    )
    insert_flight(flight, db_path=db_path)
    assign_crew_to_flight("CRW003", "FL202", "CabinCrew", db_path=db_path)

    schedule = [{
        "callsign": "FL202",
        "route": "BLR-DEL",
        "avg_duration_min": 120,
        "avg_departure_hour": 12,
        "scheduled_date": "2026-08-07",
        "scheduled_departure": "12:00",
        "scheduled_arrival": "14:00",
        "leg_type": "First Leg",
        "crew_action": "",
        "prediction": {
            "risk_level": "High",
            "expected_delay_min": 60,
            "delay_probability": 0.8,
            "factors": [],
        },
    }]

    result = proactive_crew_assignment(schedule, str(csv_path), db_path)
    rec = result["crew_recommendations"]["FL202"]

    assert rec["suggested_crew"] == {}
    assert "CabinCrew" in rec["no_standby_roles"]
    assert rec["standby_count"] == 0

    reloaded = csv_path.read_text(encoding="utf-8")
    assert "STBY-" not in reloaded
    assert reloaded.count("\n") == 2
