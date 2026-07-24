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
    "reporting time", "report time", "reporting", "report for duty",
    "sign on", "sign-on", "duty start", "duty time", "duty period",
    "check in time", "report time", "rest period", "rest requirement",
    "flying hour", "flight time limit", "minimum rest",
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
        "Rule_Query, Data_Query, Flight_Status, Schedule_Disruption, Compliance_Check, Delay_Management.\n"
        "If Schedule_Disruption, extract: scenario_flight_hours (float), scenario_is_night_duty (bool), "
        "required_counts (dict), flight_ids (list of strings).\n"
        "If Flight_Status, extract: flight_ids (list), origin (string), destination (string).\n"
        "If Compliance_Check, extract: crew_id (string), flight_ids (list).\n"
        "If Delay_Management, extract: flight_ids (list), delay_minutes (int or null), is_cancel (bool).\n"
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
        "Delay_Management": "delay",
    }
    return RouteDecision(
        intent=intent,
        route=route_map.get(intent, "data"),
        extraction=payload.get("extraction") or {},
        confidence=float(payload.get("confidence", 0.8)),
        mode="azure", raw_input=user_input,
    )


def _classify_with_groq(user_input: str) -> RouteDecision:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
    except ImportError:
        return _classify_with_regex(user_input)

    if not GROQ_API_KEY or GROQ_API_KEY == "your-groq-api-key-here":
        return _classify_with_regex(user_input)

    model = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0.0,
        request_timeout=10,
    )
    system_prompt = """You are an airline operations router. Classify user intent into ONE of these exact strings:
"Rule_Query", "Data_Query", "Flight_Status", "Schedule_Disruption", "Compliance_Check", "Delay_Management"

IMPORTANT: Handle typos and misspellings. Words like "dela", "delai", "delat", "delayy" all mean "delay". Words like "cancle", "cancel" mean "cancel".

Use "Delay_Management" for ANY delay/cancel action:
- "delay AI-301 by 2 hours", "delayed by 30 min", "dela by 7 hours", "cancel AI-501"

Use "Flight_Status" ONLY for READ-ONLY flight status queries:
- "what is the status of AI-301", "show me flight AI-301", "when does AI-301 depart"

Use "Data_Query" for ALL crew/staff/roster queries about flights or people:
- "AI-901 staff", "AI-301 crew", "who is assigned to AI-501", "show me the crew of AI-701"
- "give me information about its crew", "what staff are on AI-901"
- "who is CRW001", "show crew list", "available captains"

Use "Rule_Query" for regulations, FDTL rules, rest requirements, SOPs, policies.

Use "Compliance_Check" ONLY when validating if a NAMED crew member can fly a specific route.

Return JSON: {"intent": "...", "extraction": {}, "confidence": 0.9}"""

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ])
    try:
        response_text = response.content
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            payload = json.loads(json_match.group())
        else:
            payload = json.loads(response_text)
    except Exception:
        return _classify_with_regex(user_input)

    intent = payload.get("intent", "Data_Query")
    route_map = {
        "Rule_Query": "rag",
        "Data_Query": "data",
        "Flight_Status": "flights",
        "Schedule_Disruption": "solver",
        "Compliance_Check": "compliance",
        "Delay_Management": "delay",
    }
    return RouteDecision(
        intent=intent,
        route=route_map.get(intent, "data"),
        extraction=payload.get("extraction") or {},
        confidence=float(payload.get("confidence", 0.85)),
        mode="groq", raw_input=user_input,
    )


def route_request(user_input: str) -> Dict[str, Any]:
    result = None
    if _has_azure_credentials():
        try:
            result = asdict(_classify_with_azure(user_input))
        except Exception:
            result = None
    if result is None:
        try:
            result = asdict(_classify_with_groq(user_input))
        except Exception:
            result = None
    if result is None:
        result = asdict(_classify_with_regex(user_input))

    extraction = result.get("extraction", {})
    if not extraction.get("flight_ids"):
        regex_fids = _extract_flight_ids(user_input)
        if regex_fids:
            extraction["flight_ids"] = regex_fids
    if extraction.get("delay_minutes") is None:
        hour_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hrs|hr)", user_input, re.IGNORECASE)
        min_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minutes|min)", user_input, re.IGNORECASE)
        if hour_match:
            extraction["delay_minutes"] = int(float(hour_match.group(1)) * 60)
        elif min_match:
            extraction["delay_minutes"] = int(min_match.group(1))
    result["extraction"] = extraction
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input")
    print(json.dumps(route_request(parser.parse_args().user_input), indent=2))
