from __future__ import annotations

from datetime import datetime, timedelta

from data.models import Flight, FlightStatus, Qualification, Role


def test_role_values():
    assert Role.CAPTAIN.value == "Captain"
    assert Role.FO.value == "FO"
    assert Role.CABIN_CREW.value == "CabinCrew"


def test_qualification_validity():
    now = datetime.now()
    active = Qualification("b737", valid_from=now - timedelta(days=1), valid_until=now + timedelta(days=30))
    assert active.aircraft_type == "B737"
    assert active.is_valid()
    expired = Qualification("A320", valid_until=now - timedelta(days=1))
    assert not expired.is_valid()
    not_yet = Qualification("A321", valid_from=now + timedelta(days=1))
    assert not not_yet.is_valid()


def test_crew_is_rated(captain, sample_flight):
    assert captain.is_rated_on(sample_flight.aircraft_type)
    assert not captain.is_rated_on("A380")


def test_flight_hours(sample_flight):
    assert sample_flight.flight_hours == 2.0
    assert sample_flight.status == FlightStatus.SCHEDULED


def test_flight_sta(sample_flight):
    assert sample_flight.sta == sample_flight.std + timedelta(minutes=120)


def test_night_duty():
    night = Flight(
        flight_id="N1", origin="A", destination="B",
        std=datetime(2026, 8, 7, 23, 0), aircraft_type="B737",
    )
    day = Flight(
        flight_id="N2", origin="A", destination="B",
        std=datetime(2026, 8, 7, 12, 0), aircraft_type="B737",
    )
    assert night.is_night_duty
    assert not day.is_night_duty


def test_departs_after(sample_flight):
    assert not sample_flight.departs_after(sample_flight, min_gap_min=0)
    later = Flight(
        flight_id="FL002", origin="BLR", destination="DEL",
        std=sample_flight.sta + timedelta(minutes=90),
        aircraft_type="B737", turnaround_min=0,
    )
    assert later.departs_after(sample_flight, min_gap_min=0)
