from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

RULE_KEYWORDS = (
    "rule", "rules", "dgca", "fdtl", "regulation",
    "clause", "compliance", "legal", "sop", "policy",
    "union", "training", "recurrency", "rest requirement",
)

FLIGHT_KEYWORDS = (
    "flight", "departure", "departure", "arrive", "arriving",
    "schedule", "delay", "cancelled", "gate", "terminal",
    "upcoming", "status",
)

CREW_QUERY_KEYWORDS = (
    "crew", "pilot", "captain", "fo", "first officer",
    "cabin", "ground staff", "roster", "available", "free",
    "who are", "how many",
)

DISRUPTION_KEYWORDS = (
    "delay", "disruption", "disrupted", "cancel", "reassign",
    "replace", "cover", "gap", "need crew", "standby",
    "recovery", "replan",
)

VALIDATION_KEYWORDS = (
    "check", "validate", "legal", "can they", "eligible",
    "eligible for", "check crew", "is it legal",
)

DELAY_MANAGEMENT_KEYWORDS = (
    "delay by", "delay for", "postpone", "push back",
    "mark delayed", "mark cancelled", "cancel flight",
    "update status", "delay flight", "delayed by",
    "how does delay", "impact of delay",
)

GROUND_OPS_KEYWORDS = (
    "turnaround", "boarding", "gate event", "ground handling",
    "baggage", "ramp", "ground staff", "ground ops",
)

PASSENGER_KEYWORDS = (
    "passenger", "pax", "demand", "nationality", "connecting",
    "baggage load", "baggage profile", "baggage prediction",
    "demand by route", "passenger profile", "passenger demand",
)

STAFFING_KEYWORDS = (
    "staffing", "staff distribution", "shift coverage",
    "understaffed", "staff utilization", "shift analysis",
    "role distribution", "staff forecast", "staffing need",
)

SECURITY_KEYWORDS = (
    "security", "screening", "queue", "xray", "screen type",
    "security throughput", "queue prediction", "queue buildup",
    "security staff", "screening staff",
)

REVENUE_KEYWORDS = (
    "revenue", "retail", "spend", "duty free", "txn",
    "retail revenue", "passenger spend", "spend profile",
    "revenue by flight", "revenue by gate",
)

MAINTENANCE_KEYWORDS = (
    "maintenance", "work order", "defect", "airworthy",
    "aircraft maintenance", "maintenance impact", "maintenance log",
)

DELAY_MANAGEMENT_PATTERNS = (
    r"delay\b.*?\bby\b",
    r"delayed\b.*?\bby\b",
    r"\bcancel(?:led|led|ation)?\b",
    r"\bpostpone\b",
    r"\bpush\s*back\b",
)


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    route: str
    extraction: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    mode: str = "fallback"
    raw_input: str = ""


def _has_azure_credentials() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_API_KEY")
        and os.getenv("AZURE_OPENAI_ENDPOINT")
        and os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    )


def _contains_any(text: str, keywords: tuple) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _extract_flight_ids(user_input: str) -> list:
    pattern = r"([A-Z]{2,3}[-_]?\d{2,4})"
    return [m.upper().replace("_", "-") for m in re.findall(pattern, user_input, re.IGNORECASE)]


def _extract_flight_hours(user_input: str) -> Optional[float]:
    patterns = [
        r"(?:flight|scenario|duty|delay|disruption).*?(\d+(?:\.\d+)?)\s*(?:hour|hours|hrs|hr)",
        r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hrs|hr)",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_input, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def _extract_night_duty(user_input: str) -> bool:
    normalized = user_input.lower()
    if any(term in normalized for term in ("night", "overnight", "red-eye", "redeye")):
        return True
    if any(term in normalized for term in ("day duty", "day shift", "dayflight")):
        return False
    return False


def _extract_required_counts(user_input: str) -> Dict[str, int]:
    required = {"Captain": 1, "FO": 1, "CabinCrew": 2, "GroundStaff": 1}
    lowered = user_input.lower()
    role_patterns = [
        (r"(\d+)\s*(?:captain|captains|cpt)\b", "Captain"),
        (r"(\d+)\s*(?:fo|first officer|first officers)\b", "FO"),
        (r"(\d+)\s*(?:cabin crew|cabincrew|flight attendant)\b", "CabinCrew"),
        (r"(\d+)\s*(?:ground staff|groundstaff)\b", "GroundStaff"),
    ]
    for pattern, role in role_patterns:
        match = re.search(pattern, lowered)
        if match:
            required[role] = int(match.group(1))
    return required


def _classify_with_regex(user_input: str) -> RouteDecision:
    if _contains_any(user_input, RULE_KEYWORDS):
        return RouteDecision(
            intent="Rule_Query", route="rag",
            confidence=0.7, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, VALIDATION_KEYWORDS):
        return RouteDecision(
            intent="Compliance_Check", route="compliance",
            confidence=0.7, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, SECURITY_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Security_Analytics", route="security",
            extraction=extraction,
            confidence=0.75, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, REVENUE_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Revenue_Analytics", route="revenue",
            extraction=extraction,
            confidence=0.75, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, MAINTENANCE_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Maintenance_Status", route="maintenance",
            extraction=extraction,
            confidence=0.7, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, PASSENGER_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Passenger_Flow", route="passenger",
            extraction=extraction,
            confidence=0.75, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, STAFFING_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Staff_Analytics", route="staffing",
            extraction=extraction,
            confidence=0.75, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, GROUND_OPS_KEYWORDS):
        extraction = {"flight_ids": _extract_flight_ids(user_input)}
        return RouteDecision(
            intent="Turnaround_Status", route="ground_ops",
            extraction=extraction,
            confidence=0.75, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, DELAY_MANAGEMENT_KEYWORDS) or any(re.search(p, user_input, re.IGNORECASE) for p in DELAY_MANAGEMENT_PATTERNS):
        delay_minutes = None
        hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hrs|hr)", user_input, re.IGNORECASE)
        min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minutes|min)", user_input, re.IGNORECASE)
        if hour_match:
            delay_minutes = int(float(hour_match.group(1)) * 60)
        elif min_match:
            delay_minutes = int(min_match.group(1))

        is_cancel = any(w in user_input.lower() for w in ("cancel", "cancelled", "cancellation"))

        extraction = {"flight_ids": _extract_flight_ids(user_input), "delay_minutes": delay_minutes, "is_cancel": is_cancel}
        return RouteDecision(
            intent="Delay_Management", route="delay",
            extraction=extraction,
            confidence=0.8, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, DISRUPTION_KEYWORDS):
        flight_ids = _extract_flight_ids(user_input)
        extraction: Dict[str, Any] = {}
        hours = _extract_flight_hours(user_input)
        if hours:
            extraction["scenario_flight_hours"] = hours
        extraction["scenario_is_night_duty"] = _extract_night_duty(user_input)
        extraction["required_counts"] = _extract_required_counts(user_input)
        if flight_ids:
            extraction["flight_ids"] = flight_ids
        return RouteDecision(
            intent="Schedule_Disruption", route="solver",
            extraction=extraction,
            confidence=0.7, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, CREW_QUERY_KEYWORDS):
        return RouteDecision(
            intent="Data_Query", route="data",
            confidence=0.65, mode="regex_fallback", raw_input=user_input,
        )
    if _contains_any(user_input, FLIGHT_KEYWORDS):
        return RouteDecision(
            intent="Flight_Status", route="flights",
            confidence=0.65, mode="regex_fallback", raw_input=user_input,
        )

    flight_ids = _extract_flight_ids(user_input)
    if flight_ids:
        return RouteDecision(
            intent="Flight_Status", route="flights",
            extraction={"flight_ids": flight_ids},
            confidence=0.6, mode="regex_fallback", raw_input=user_input,
        )

    return RouteDecision(
        intent="Data_Query", route="data",
        confidence=0.5, mode="regex_fallback", raw_input=user_input,
    )


def _classify_with_azure(user_input: str) -> RouteDecision:
    from langchain_openai import AzureChatOpenAI
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0,
    )
    prompt = (
        "You are an airline operations router. Classify the user's intent into ONE of: "
        "Rule_Query, Data_Query, Flight_Status, Schedule_Disruption, Compliance_Check, Recovery_Plan, Delay_Management, "
        "Staff_Analytics, Passenger_Flow, Turnaround_Status, Security_Analytics, Revenue_Analytics, Maintenance_Status.\n"
        "If Schedule_Disruption, extract: scenario_flight_hours (float), scenario_is_night_duty (bool), "
        "required_counts (dict), flight_ids (list of strings).\n"
        "If Flight_Status, extract: flight_ids (list), origin (string), destination (string).\n"
        "If Compliance_Check, extract: crew_id (string), flight_ids (list).\n"
        "If Delay_Management, extract: flight_ids (list), delay_minutes (int or null), is_cancel (bool).\n"
        "If Staff_Analytics, Passenger_Flow, Turnaround_Status, Security_Analytics, Revenue_Analytics, or Maintenance_Status, "
        "extract: flight_ids (list if mentioned).\n"
        "Return ONLY valid JSON with keys: intent, extraction, confidence.\n\n"
        f"User input: {user_input}"
    )
    try:
        payload = json.loads(getattr(model.invoke(prompt), "content", "{}"))
    except Exception:
        return _classify_with_regex(user_input)

    intent = payload.get("intent", "Data_Query")
    route_map = {
        "Rule_Query": "rag",
        "Data_Query": "data",
        "Flight_Status": "flights",
        "Schedule_Disruption": "solver",
        "Compliance_Check": "compliance",
        "Recovery_Plan": "recovery",
        "Delay_Management": "delay",
        "Staff_Analytics": "staffing",
        "Passenger_Flow": "passenger",
        "Turnaround_Status": "ground_ops",
        "Security_Analytics": "security",
        "Revenue_Analytics": "revenue",
        "Maintenance_Status": "maintenance",
    }
    return RouteDecision(
        intent=intent,
        route=route_map.get(intent, "data"),
        extraction=payload.get("extraction", {}),
        confidence=float(payload.get("confidence", 0.8)),
        mode="azure", raw_input=user_input,
    )


def _classify_with_ollama(user_input: str) -> RouteDecision:
    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return _classify_with_regex(user_input)

    model = ChatOllama(model="qwen2.5:3b", temperature=0.0, format="json")
    system_prompt = """You are an airline operations router. Classify the user's intent into ONE of these:
    1. "Rule_Query": Questions about DGCA rules, regulations, SOPs, policies, or compliance requirements.
    2. "Data_Query": Questions about crew roster, availability, who is free, crew counts, etc.
    3. "Flight_Status": Questions about flight schedules, departures, arrivals, delays, or cancellations.
    4. "Schedule_Disruption": Requests to assign crew for a disrupted/cancelled flight requiring new crew.
    5. "Compliance_Check": Requests to validate if a specific crew member can fly a specific route.
    6. "Recovery_Plan": Major disruptions needing full replanning of multiple flights.
    7. "Delay_Management": Requests to delay a flight by X hours/minutes, or cancel a flight, or check impact of a delay on assigned crew.
       For Delay_Management, extract: "flight_ids" (list), "delay_minutes" (int or null), "is_cancel" (bool).
    8. "Staff_Analytics": Questions about staff distribution, shift coverage, understaffing, or staff utilization.
    9. "Passenger_Flow": Questions about passenger demand, baggage load, connecting passengers, or passenger profiles.
    10. "Turnaround_Status": Questions about turnaround times, boarding efficiency, ground ops, or gate events.
    11. "Security_Analytics": Questions about security throughput, screening queues, or security staff performance.
    12. "Revenue_Analytics": Questions about retail revenue, passenger spend, duty free, or retail demand.
    13. "Maintenance_Status": Questions about maintenance logs, work orders, defects, or aircraft airworthiness.

    For Schedule_Disruption, extract: "scenario_flight_hours" (float), "scenario_is_night_duty" (bool),
    "required_counts" (dict of role:count), "flight_ids" (list of strings if mentioned).
    For Flight_Status, extract: "flight_ids" (list), "origin" (string), "destination" (string).
    For Compliance_Check, extract: "crew_id" (string), "flight_ids" (list).
    For Staff_Analytics, Passenger_Flow, Turnaround_Status, Security_Analytics, Revenue_Analytics, or Maintenance_Status,
    extract: "flight_ids" (list if mentioned).
    Return ONLY JSON with keys: "intent", "extraction", and "confidence"."""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])
    try:
        payload = json.loads(response.content)
    except Exception:
        return _classify_with_regex(user_input)

    intent = payload.get("intent", "Data_Query")
    route_map = {
        "Rule_Query": "rag",
        "Data_Query": "data",
        "Flight_Status": "flights",
        "Schedule_Disruption": "solver",
        "Compliance_Check": "compliance",
        "Recovery_Plan": "recovery",
        "Delay_Management": "delay",
        "Staff_Analytics": "staffing",
        "Passenger_Flow": "passenger",
        "Turnaround_Status": "ground_ops",
        "Security_Analytics": "security",
        "Revenue_Analytics": "revenue",
        "Maintenance_Status": "maintenance",
    }
    return RouteDecision(
        intent=intent,
        route=route_map.get(intent, "data"),
        extraction=payload.get("extraction", {}),
        confidence=float(payload.get("confidence", 0.85)),
        mode="ollama", raw_input=user_input,
    )


def route_request(user_input: str) -> Dict[str, Any]:
    if _has_azure_credentials():
        try:
            return asdict(_classify_with_azure(user_input))
        except Exception:
            pass
    try:
        return asdict(_classify_with_ollama(user_input))
    except Exception:
        pass
    return asdict(_classify_with_regex(user_input))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")
    print(json.dumps(route_request(parser.parse_args().user_input), indent=2))
