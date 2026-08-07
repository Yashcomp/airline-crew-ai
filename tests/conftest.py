from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.models import CrewMember, Qualification, Role  # noqa: E402


@pytest.fixture
def crew_csv(tmp_path):
    """A small offline crew roster in the same format as crew_standby_list.csv."""
    path = tmp_path / "crew_standby_list.csv"
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
        "CRW005,Dev Patel,RampAgent,0.0,2.0,0,Legal,45.0,1.0,,BLR,2,0.0,1,0,",
        "CRW006,Anita Das,BaggageHandler,0.0,2.0,0,Legal,45.0,1.0,,BLR,2,0.0,1,0,",
        "CRW007,Suresh Iyer,CabinCleaner,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
        "CRW008,Priya Menon,CheckinAgent,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
        "CRW009,Karan Joshi,SecurityAgent,0.0,2.0,0,Legal,40.0,1.0,,BLR,2,0.0,1,0,",
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def db_path(tmp_path):
    """An isolated SQLite database with all app tables initialised."""
    from data.flights_db import init_db
    from data.opensky_db import init_opensky_tables

    path = init_db(db_path=tmp_path / "flights.db")
    init_opensky_tables(path)
    return path


@pytest.fixture
def captain():
    return CrewMember(
        crew_id="CAP1",
        name="Test Captain",
        role=Role.CAPTAIN,
        current_duty_hours=2.0,
        rolling_7_day_hours=20.0,
        rest_status="Legal",
        qualifications=[Qualification("B737")],
        base_airport="BLR",
    )


@pytest.fixture
def sample_flight():
    from datetime import datetime, timezone

    from data.models import Flight

    return Flight(
        flight_id="FL001",
        origin="BLR",
        destination="DEL",
        std=datetime(2026, 8, 7, 10, 0, tzinfo=timezone.utc),
        aircraft_type="B737",
        flight_duration_min=120,
        pax_count=150,
    )
