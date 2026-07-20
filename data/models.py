from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(str, Enum):
    CAPTAIN = "Captain"
    FO = "FO"
    CABIN_CREW = "CabinCrew"
    GROUND_STAFF = "GroundStaff"
    SECURITY = "Security"
    GROUND_HANDLING = "GroundHandling"
    MAINTENANCE = "Maintenance"
    OPERATIONS = "Operations"
    CABIN_CLEANING = "CabinCleaning"
    RETAIL = "Retail"


class FlightStatus(str, Enum):
    SCHEDULED = "scheduled"
    DELAYED = "delayed"
    BOARDING = "boarding"
    DEPARTED = "departed"
    LANDED = "landed"
    CANCELLED = "cancelled"
    DIVERTED = "diverted"


class Qualification:
    def __init__(
        self,
        aircraft_type: str,
        valid_from: Optional[datetime] = None,
        valid_until: Optional[datetime] = None,
    ):
        self.aircraft_type = aircraft_type.upper()
        self.valid_from = valid_from
        self.valid_until = valid_until

    def is_valid(self, on_date: Optional[datetime] = None) -> bool:
        check = on_date or datetime.now()
        if self.valid_from and check < self.valid_from:
            return False
        if self.valid_until and check > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aircraft_type": self.aircraft_type,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }


@dataclass
class CrewMember:
    crew_id: str
    name: str
    role: Role
    current_duty_hours: float = 0.0
    rolling_7_day_hours: float = 0.0
    consecutive_night_shifts: int = 0
    rest_status: str = "Legal"
    base_cost: float = 1.0
    overtime_multiplier: float = 1.0
    qualifications: List[Qualification] = field(default_factory=list)
    base_airport: str = ""
    seniority: int = 0
    hours_flown_30_days: float = 0.0
    days_since_rest: int = 0
    consecutive_days_on: int = 0

    @property
    def rated_aircraft(self) -> List[str]:
        return [q.aircraft_type for q in self.qualifications]

    def is_rated_on(self, aircraft_type: str, on_date: Optional[datetime] = None) -> bool:
        check = on_date or datetime.now()
        return any(
            q.aircraft_type == aircraft_type.upper() and q.is_valid(check)
            for q in self.qualifications
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crew_id": self.crew_id,
            "name": self.name,
            "role": self.role.value,
            "current_duty_hours": self.current_duty_hours,
            "rolling_7_day_hours": self.rolling_7_day_hours,
            "consecutive_night_shifts": self.consecutive_night_shifts,
            "rest_status": self.rest_status,
            "base_cost": self.base_cost,
            "overtime_multiplier": self.overtime_multiplier,
            "qualifications": [q.to_dict() for q in self.qualifications],
            "base_airport": self.base_airport,
            "seniority": self.seniority,
            "hours_flown_30_days": self.hours_flown_30_days,
            "days_since_rest": self.days_since_rest,
            "consecutive_days_on": self.consecutive_days_on,
        }


@dataclass
class Flight:
    flight_id: str
    origin: str
    destination: str
    std: datetime
    aircraft_type: str
    status: FlightStatus = FlightStatus.SCHEDULED
    gate: Optional[str] = None
    terminal: Optional[str] = None
    pax_count: int = 0
    turnaround_min: int = 45
    flight_duration_min: int = 120
    is_international: bool = False
    disruption_reason: Optional[str] = None

    @property
    def sta(self) -> datetime:
        return self.std + timedelta(minutes=self.flight_duration_min)

    @property
    def flight_hours(self) -> float:
        return round(self.flight_duration_min / 60.0, 2)

    @property
    def is_night_duty(self) -> bool:
        return self.std.hour >= 22 or self.std.hour < 5

    def departs_after(self, other: "Flight", min_gap_min: int = 0) -> bool:
        return self.std >= other.sta + timedelta(minutes=min_gap_min + self.turnaround_min)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flight_id": self.flight_id,
            "origin": self.origin,
            "destination": self.destination,
            "std": self.std.isoformat(),
            "sta": self.sta.isoformat(),
            "aircraft_type": self.aircraft_type,
            "status": self.status.value,
            "gate": self.gate,
            "terminal": self.terminal,
            "pax_count": self.pax_count,
            "turnaround_min": self.turnaround_min,
            "flight_duration_min": self.flight_duration_min,
            "flight_hours": self.flight_hours,
            "is_night_duty": self.is_night_duty,
            "is_international": self.is_international,
            "disruption_reason": self.disruption_reason,
        }


@dataclass
class DisruptionEvent:
    event_id: str
    event_type: str
    affected_flight_ids: List[str]
    severity: str
    timestamp: datetime
    description: str = ""
    estimated_delay_min: int = 0
    requires_replacement_crew: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "affected_flight_ids": self.affected_flight_ids,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "estimated_delay_min": self.estimated_delay_min,
            "requires_replacement_crew": self.requires_replacement_crew,
        }


@dataclass
class OpsStaff:
    staff_id: str
    name: str
    department: str = ""
    role: Role = Role.GROUND_STAFF
    terminal: str = ""
    valid_until: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "staff_id": self.staff_id,
            "name": self.name,
            "department": self.department,
            "role": self.role.value,
            "terminal": self.terminal,
            "valid_until": self.valid_until,
        }


@dataclass
class ShiftRecord:
    staff_id: str
    shift_date: str = ""
    shift_start: str = ""
    shift_end: str = ""
    terminal: str = ""
    gate: str = ""
    shift_hours: float = 0.0
    overtime: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "staff_id": self.staff_id,
            "shift_date": self.shift_date,
            "shift_start": self.shift_start,
            "shift_end": self.shift_end,
            "terminal": self.terminal,
            "gate": self.gate,
            "shift_hours": self.shift_hours,
            "overtime": self.overtime,
        }


@dataclass
class BaggageRecord:
    baggage_tag: str
    pnr_code: str = ""
    flight_id: str = ""
    passenger_id: str = ""
    weight: float = 0.0
    dimensions: str = ""
    stage: str = ""
    gate: str = ""
    checkin_time: str = ""
    loaded_time: str = ""
    pieces: int = 0
    status: str = ""
    delay: bool = False
    delay_mins: int = 0
    assigned_staff: str = ""
    assigned_at: str = ""
    is_oversized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baggage_tag": self.baggage_tag,
            "flight_id": self.flight_id,
            "passenger_id": self.passenger_id,
            "weight": self.weight,
            "pieces": self.pieces,
            "status": self.status,
            "delay_mins": self.delay_mins,
            "stage": self.stage,
        }


@dataclass
class PassengerRecord:
    pnr_code: str = ""
    ticket_no: str = ""
    passenger_id: str = ""
    first_name: str = ""
    last_name: str = ""
    nationality: str = ""
    dob: str = ""
    gender: str = ""
    seat: str = ""
    seat_class: str = ""
    flight_id: str = ""
    booking_time: str = ""
    boarding_time: str = ""
    gate: str = ""
    baggage_count: int = 0
    upgrade: bool = False
    loyalty_score: float = 0.0
    is_connecting: bool = False
    fare_class: str = ""
    fare_amount: float = 0.0
    age_category: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passenger_id": self.passenger_id,
            "full_name": self.full_name,
            "flight_id": self.flight_id,
            "nationality": self.nationality,
            "seat": self.seat,
            "seat_class": self.seat_class,
            "baggage_count": self.baggage_count,
            "loyalty_score": self.loyalty_score,
            "is_connecting": self.is_connecting,
            "age_category": self.age_category,
        }


@dataclass
class SecurityScreening:
    screen_id: str = ""
    passenger_id: str = ""
    ticket_code: str = ""
    screen_type: int = 0
    queue_start: str = ""
    queue_end: str = ""
    clearance_time: str = ""
    result: str = ""
    alarm_type: str = ""
    secondary: bool = False
    staff_id: str = ""
    xray_machine: str = ""
    processing_time: float = 0.0
    has_alarm: bool = False
    has_secondary: bool = False
    shift_id: str = ""
    pass_per_hour: int = 0
    capacity: int = 0
    utilization: float = 0.0
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "screen_id": self.screen_id,
            "passenger_id": self.passenger_id,
            "screen_type": self.screen_type,
            "result": self.result,
            "processing_time": self.processing_time,
            "staff_id": self.staff_id,
            "has_alarm": self.has_alarm,
        }


@dataclass
class RetailTransaction:
    txn_id: str = ""
    staff_id: str = ""
    store_name: str = ""
    store_type: str = ""
    passenger_id: str = ""
    flight_id: str = ""
    txn_time: str = ""
    product: str = ""
    quantity: int = 0
    amount: float = 0.0
    discount: float = 0.0
    payment_method: str = ""
    currency: str = ""
    terminal: str = ""
    location: str = ""
    is_connecting: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "flight_id": self.flight_id,
            "product": self.product,
            "amount": self.amount,
            "terminal": self.terminal,
        }


@dataclass
class MaintenanceLog:
    work_order: str = ""
    aircraft_reg: str = ""
    flight_id: str = ""
    task_type: str = ""
    staff_id: str = ""
    assigned_at: str = ""
    resolved_at: str = ""
    severity: int = 0
    duration_hrs: float = 0.0
    defect: str = ""
    component: str = ""
    impact_level: int = 0
    resolved_by: str = ""
    airworthy: bool = False
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_order": self.work_order,
            "aircraft_reg": self.aircraft_reg,
            "flight_id": self.flight_id,
            "task_type": self.task_type,
            "severity": self.severity,
            "defect": self.defect,
            "component": self.component,
            "airworthy": self.airworthy,
        }
