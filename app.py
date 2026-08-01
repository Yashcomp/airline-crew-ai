from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rag_engine import retrieve_legal_guidance
from router import route_request, _is_plain_greeting
from solver import solve_from_csv, solve_multi_flight
from data.flights_db import (
    init_db, get_flights, get_flight,
    get_flight_stats, get_crew_for_flight,
    assign_crew_to_flight, unassign_crew_from_flight,
)
from data.crew_loader import load_crew
from data.models import FlightStatus
from agents.flight_agent import answer_flight_query, answer_crew_query
from agents.compliance_agent import validate_single_crew, get_at_risk_crew
from ml_engine.resource_augmenter import forecast_crew_needs
from data.delay_handler import process_delay, process_cancellation, find_replacement_crew
from data.delay_handler import analyze_delay_impact, proactive_crew_assignment
from data.opensky_db import (
    poll_live_data, get_today_schedule, get_schedule_for_date, update_daily_data,
    get_model_flights_with_status, sync_opensky_flights_to_db,
    log_predictions, backfill_predictions, update_actuals,
    get_prediction_audit, compute_audit_metrics,
    get_flight_actuals_for_date, get_data_date_range,
)
from data.staff_manager import REQUIRED_CREW
from validators.dgca_validator import GROUND_ROLES, check_crew_eligibility

APP_TITLE = "Airline Crew Operations Hub"
DEFAULT_CSV_PATH = Path(__file__).parent / "crew_standby_list.csv"
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "flights.db"


def _get_crew_shift_lookup(csv_path=None):
    path = csv_path or str(DEFAULT_CSV_PATH)
    crew = load_crew(path)
    return {m.crew_id: m.shift for m in crew if m.shift}

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Flight scheduling, crew management, DGCA compliance, and disruption recovery — all in one place.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_initialized" not in st.session_state:
    init_db(DEFAULT_DB_PATH)
    st.session_state.db_initialized = True

if "system_initialized" not in st.session_state:
    try:
        from data.opensky_db import _connect as _opensky_connect
        _conn = _opensky_connect(DEFAULT_DB_PATH)
        _flight_count = _conn.execute("SELECT COUNT(*) FROM opensky_flights").fetchone()[0]
        _model_path = Path(__file__).parent / "ml_engine" / "models" / "delay_classifier.pkl"
        _model_exists = _model_path.exists()
        st.session_state["system_initialized"] = _flight_count > 0 and _model_exists
        st.session_state["flight_count"] = _flight_count
        _conn.close()
    except Exception:
        st.session_state["system_initialized"] = False
        st.session_state["flight_count"] = 0


def _build_chat_context() -> str:
    lines = []
    try:
        from data.opensky_db import get_flight_schedule, SCHEDULE_LIMIT
        schedule = get_flight_schedule(limit=SCHEDULE_LIMIT, db_path=DEFAULT_DB_PATH)
        if schedule:
            lines.append("TODAY'S BLR FLIGHTS (%d):" % len(schedule))
            for f in schedule:
                lines.append(
                    "  %s %s->%s dep %02d:%02d %dmin delay_rate=%d%%" % (
                        f["callsign"], f["origin"], f["destination"],
                        f["avg_departure_hour"], f["avg_departure_minute"],
                        f["avg_duration_min"], f["delay_rate_pct"],
                    )
                )
    except Exception:
        pass

    try:
        from data.crew_loader import load_crew
        crew = load_crew(str(DEFAULT_CSV_PATH))
        if crew:
            role_counts = {}
            for c in crew:
                r = c.role.value
                role_counts[r] = role_counts.get(r, 0) + 1
            lines.append("")
            lines.append("CREW ROSTER: %d total (%s)" % (
                len(crew),
                ", ".join("%s=%d" % (r, n) for r, n in sorted(role_counts.items()))
            ))
    except Exception:
        pass

    try:
        from data.flights_db import get_flights, get_crew_for_flight
        flights = get_flights(db_path=DEFAULT_DB_PATH)
        if flights:
            assigned_ids = set()
            flight_summary = []
            for f in flights:
                assigned = get_crew_for_flight(f.flight_id, DEFAULT_DB_PATH)
                n = len(assigned)
                for a in assigned:
                    assigned_ids.add(a["crew_id"])
                flight_summary.append("%s=%d" % (f.flight_id, n))
            lines.append("CREW ASSIGNMENTS: %s" % ", ".join(flight_summary))
            lines.append("ASSIGNED: %d crew | STANDBY: %d crew" % (
                len(assigned_ids), len(crew) - len(assigned_ids)
            ))
    except Exception:
        pass

    try:
        from data.opensky_db import get_schedule_for_date as _gfd
        from datetime import date as _d
        _today_schedule = _gfd(_d.today(), db_path=DEFAULT_DB_PATH)
        _risk_flights = [f for f in _today_schedule if f["prediction"]["risk_level"] in ("High", "Medium")]
        if _risk_flights:
            lines.append("")
            lines.append("TODAY'S AT-RISK FLIGHTS (%d):" % len(_risk_flights))
            for f in _risk_flights:
                p = f["prediction"]
                lines.append(
                    "  %s %s %s risk=%s prob=%.0f%% delay=%dmin" % (
                        f["callsign"], f["route"], f["scheduled_departure"],
                        p["risk_level"], p["delay_probability"] * 100,
                        p["expected_delay_min"],
                    )
                )
    except Exception:
        pass

    return "\n".join(lines)


def _answer_planning_query(user_input: str, extraction: dict = None) -> str:
    from datetime import date as date_type
    from data.opensky_db import get_schedule_for_date

    extraction = extraction or {}
    today = date_type.today()
    target_date = None

    raw_date = extraction.get("target_date")
    if raw_date:
        try:
            target_date = date_type.fromisoformat(raw_date)
        except Exception:
            pass

    if target_date is None:
        user_lower = user_input.lower()
        if "tomorrow" in user_lower:
            target_date = today + timedelta(days=1)
        elif "today" in user_lower:
            target_date = today
        else:
            try:
                from dateutil import parser as dateutil_parser
                parsed = dateutil_parser.parse(
                    user_input, fuzzy=True,
                    default=datetime.now(),
                )
                target_date = parsed.date()
            except Exception:
                pass

    if target_date is None:
        target_date = today

    max_date = today + timedelta(days=16)
    if target_date > max_date:
        target_date = max_date
    if target_date < today:
        target_date = today

    try:
        schedule = get_schedule_for_date(target_date, db_path=DEFAULT_DB_PATH)
    except Exception as e:
        return f"Failed to predict delays for {target_date}: {e}"

    risk_flights = [f for f in schedule if f["prediction"]["risk_level"] in ("High", "Medium")]

    day_name = target_date.strftime("%A, %B %d, %Y")
    if not risk_flights:
        return f"No high or medium risk flights predicted for **{day_name}**."

    lines = [f"### Flights at Risk — {day_name}\n"]
    lines.append("| Flight | Route | Departure | Risk | Delay Prob | Expected Delay | Factors |")
    lines.append("|--------|-------|-----------|------|------------|----------------|---------|")
    for f in risk_flights:
        p = f["prediction"]
        factors = "; ".join(p["factors"][:2]) if p["factors"] else "-"
        lines.append(
            "| %s | %s | %s | **%s** | %.0f%% | %.0f min | %s |" % (
                f["callsign"], f["route"], f["scheduled_departure"],
                p["risk_level"], p["delay_probability"] * 100,
                p["expected_delay_min"], factors,
            )
        )

    high = sum(1 for f in risk_flights if f["prediction"]["risk_level"] == "High")
    med = sum(1 for f in risk_flights if f["prediction"]["risk_level"] == "Medium")
    low = sum(1 for f in schedule if f["prediction"]["risk_level"] == "Low")
    lines.append(
        "\n**%d** at-risk flights (%d High, %d Medium) out of **%d** total (%d Low)."
        % (len(risk_flights), high, med, len(schedule), low)
    )

    return "\n".join(lines)


def _groq_rag_chat(question: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
    from rag_engine import retrieve_legal_guidance
    from router import _is_plain_greeting

    context = _build_chat_context()

    dgca_context = ""
    if not _is_plain_greeting(question):
        try:
            dgca_context = retrieve_legal_guidance(question)
        except Exception:
            pass

    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0.1,
        request_timeout=15,
    )

    system_prompt = (
        "You are an airline operations assistant for Bangalore (VOBL) airport.\n"
        "Answer questions accurately based on the provided data. Be concise and direct.\n"
        "Use tables when presenting flight or crew data.\n\n"
        "%s\n\n" % context
    )

    if dgca_context:
        system_prompt += "RELEVANT DGCA RULES:\n%s\n\n" % dgca_context

    system_prompt += (
        "\nWhen asked about flight delays or risk on a specific future date, "
        "you have access to ML delay predictions for any date up to 16 days ahead. "
        "The system can look up schedules, delay probabilities, weather forecasts, "
        "and crew assignments for any date. Ask the user for a specific date if needed.\n"
        "When asked about crew assignment for a date, the system can suggest DGCA-compliant "
        "crew based on standby roster, duty hours, and rest status.\n"
        "When asked about delays, check the assigned crew's duty hours and rolling 7-day hours "
        "against DGCA limits (Captain/FO: 12h duty, 35h rolling; CabinCrew: 14h duty, 45h rolling; "
        "Ground staff (RampAgent/BaggageHandler/CabinCleaner/CheckinAgent/SecurityAgent): 10h duty, no rolling limit). "
        "Flag any violations and suggest replacements from the standby crew list.\n"
        "When asked about a specific flight, provide its route, schedule, delay risk, and assigned crew.\n"
        "When asked about crew availability, list crew members who are LEGAL (rested) and not assigned to any flight."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        return response.content
    except Exception as e:
        return "Error connecting to AI service: %s" % str(e)


def _groq_general_chat(question: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL

    llm = ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=0.7,
        request_timeout=15,
    )

    system_prompt = (
        "You are a friendly, helpful airline operations assistant for Bangalore (VOBL) airport.\n"
        "You can chat normally about general topics and greet the user.\n"
        "If they mention flights, crew, delays, rules, or dates, suggest they ask a specific "
        "operational question (e.g. a flight number, a crew member, or a date) and you can look it up.\n"
        "Keep replies concise, warm, and natural."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question),
        ])
        return response.content
    except Exception:
        return "Hi! I'm your airline ops assistant for VOBL. Ask me about flights, crew, delays, DGCA rules, or predictions and I'll help you out."


# === 5 MAIN TABS ===
tab_chat, tab_ops, tab_forecast, tab_live, tab_planning = st.tabs([
    "Chat", "Crew", "Forecasting", "Live Tracking", "Planning",
])

st.markdown(
    """
    <style>
    [data-testid='stChatMessage']{scroll-margin-bottom:6.5rem}
    :root:has([data-testid='stTab'][aria-selected='true']:not([id='0'])) [data-testid='stBottom']{display:none !important}
    </style>
    """,
    unsafe_allow_html=True,
)

prompt = None
with st.bottom:
    prompt = st.chat_input("Ask about flights, crew, rules, or disruptions...")


# ============================================================
# TAB 1: CHAT
# ============================================================
with tab_chat:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                msg_type = message.get("type")
                if msg_type == "rule":
                    st.markdown("### Rule Analysis")
                    st.info(message["content"])
                    if "category" in message:
                        st.caption(f"Category: {message['category']}")
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

                elif msg_type == "data":
                    st.markdown("### Crew Roster Analysis")
                    st.write(message["content"])
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

                elif msg_type == "chat":
                    st.markdown("### AI Assistant")
                    st.write(message["content"])
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

                elif msg_type == "solver":
                    st.markdown("### Operational Action Plan")
                    st.write(message["content"])
                    col1, col2 = st.columns(2)
                    col1.metric("Total Cost", f"${message['total_cost']:,.2f}")
                    col2.metric("Crew Selected", message["selected_count"])
                    df = pd.DataFrame(message["selected_crew"])
                    if not df.empty:
                        display_cols = [c for c in df.columns if not c.startswith("qualifications")]
                        st.dataframe(df[display_cols], hide_index=True)
                        st.download_button(
                            "Download Assignment (CSV)",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="crew_assignment.csv",
                            mime="text/csv",
                            key=f"dl_{message.get('id', 0)}",
                        )

                elif msg_type == "flights":
                    st.markdown("### Flight Information")
                    st.write(message["content"])
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

                elif msg_type == "compliance":
                    st.markdown("### Compliance Check")
                    st.write(message["content"])
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

                elif msg_type == "multi_solver":
                    st.markdown("### Multi-Flight Recovery Plan")
                    st.write(message["content"])
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Cost", f"${message['total_cost']:,.2f}")
                    col2.metric("Crew Selected", message["selected_count"])
                    col3.metric("Flights Covered", message.get("flight_count", 0))
                    df = pd.DataFrame(message["selected_crew"])
                    if not df.empty:
                        display_cols = [c for c in df.columns if not c.startswith("qualifications")]
                        st.dataframe(df[display_cols], hide_index=True)

                elif msg_type == "planning":
                    st.markdown("### Flight Risk Forecast")
                    st.write(message["content"])
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(message["decision"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                if "pending_delay" in st.session_state:
                    pending = st.session_state.pending_delay
                    user_lower = prompt.strip().lower()
                    yes_words = {"yes", "confirm", "go ahead", "proceed", "ok", "okay", "do it", "yep", "yeah", "sure", "y"}
                    no_words = {"no", "cancel", "never mind", "nope", "nah", "n", "stop"}

                    if user_lower in yes_words:
                        del st.session_state.pending_delay
                        fid = pending["flight_id"]
                        delay_min = pending["delay_minutes"]

                        delay_result = process_delay(fid, delay_min, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                        roster_scan = find_replacement_crew(fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH, use_llm=True)

                        msg = f"**Flight {fid} delayed +{int(delay_min)} min**\n\n"

                        if delay_result.get("unassigned_count", 0) > 0:
                            msg += f"{delay_result['unassigned_count']} crew removed (DGCA violation).\n\n"

                        if roster_scan["status"] == "success":
                            from data.staff_manager import REQUIRED_CREW
                            assigned_after = get_crew_for_flight(fid, DEFAULT_DB_PATH)
                            assigned_role_counts = {}
                            for a in assigned_after:
                                r = a["role"]
                                assigned_role_counts[r] = assigned_role_counts.get(r, 0) + 1

                            missing = {}
                            for role_name, req in REQUIRED_CREW.items():
                                have = assigned_role_counts.get(role_name, 0)
                                if have < req:
                                    missing[role_name] = req - have

                            assigned_replacements = []
                            if missing:
                                for c in roster_scan.get("eligible_standby", []):
                                    r_name = c.get("role")
                                    if r_name in missing and missing[r_name] > 0:
                                        ar = assign_crew_to_flight(c["crew_id"], fid, r_name, DEFAULT_DB_PATH)
                                        if ar.get("status") == "success":
                                            assigned_replacements.append({"name": c["name"], "role": r_name})
                                            missing[r_name] -= 1

                            if assigned_replacements:
                                msg += "**Replacements assigned:**\n"
                                for r in assigned_replacements:
                                    msg += f"- {r['name']} ({r['role']})\n"
                            else:
                                msg += "No replacements needed.\n"

                        st.markdown("### Delay Executed")
                        st.write(msg)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "flights",
                            "content": msg, "decision": {},
                        })
                        st.stop()

                    elif user_lower in no_words:
                        del st.session_state.pending_delay
                        msg = "Delay cancelled. No changes made."
                        st.write(msg)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "flights",
                            "content": msg, "decision": {},
                        })
                        st.stop()

                decision = route_request(prompt)
                intent = decision.get("intent", "Data_Query")

                if intent in ("Rule_Query", "Data_Query", "Flight_Status", "Planning_Query") and _is_plain_greeting(prompt):
                    decision["intent"] = "General_Chat"
                    decision["route"] = "chat"
                    decision["extraction"] = {}
                    intent = "General_Chat"

                delay_override_patterns = [
                    r"del[ao].*?\bby\b",        # "dela by", "delao by"
                    r"del[ao].*?\bfor\b",       # "dela for"
                    r"del[ao].*?\b\d+\s*(?:h|hr)", # "dela 7h", "dela 7hr"
                    r"\bdelay\b.*?\bby\b",       # "delay by"
                    r"\bdelayed\b.*?\bby\b",     # "delayed by"
                    r"\bdelay\b.*?\bfor\b",      # "delay for"
                    r"\bdelay\b.*?\b\d+\s*(?:h|hr)", # "delay 7h"
                    r"\bcancel(?:led|ed)?\b",    # "cancel", "cancelled"
                ]
                is_delay_override = intent != "Delay_Management" and any(re.search(p, prompt, re.IGNORECASE) for p in delay_override_patterns)
                if is_delay_override:
                    delay_minutes_override = None
                    hour_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hours|hrs|hr|h)\b", prompt, re.IGNORECASE)
                    min_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:min|minutes)", prompt, re.IGNORECASE)
                    if hour_m:
                        delay_minutes_override = int(float(hour_m.group(1)) * 60)
                    elif min_m:
                        delay_minutes_override = int(min_m.group(1))
                    fids_override = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", prompt, re.IGNORECASE)
                    decision["intent"] = "Delay_Management"
                    decision["extraction"]["flight_ids"] = fids_override
                    decision["extraction"]["delay_minutes"] = delay_minutes_override
                    decision["extraction"]["is_cancel"] = any(re.search(p, prompt, re.IGNORECASE) for p in [r"\bcancel(?:led|ed)?\b"]) and delay_minutes_override is None
                    intent = "Delay_Management"

                crew_override_patterns = [
                    r"[A-Z]{2,3}[-_]?\d{2,4}\s+(?:staff|crew|team)",
                    r"(?:staff|crew|team)\s+(?:of|for|on|assigned)\s+[A-Z]{2,3}[-_]?\d{2,4}",
                    r"its\s+(?:staff|crew|team)",
                    r"(?:who|what)\s+(?:is|are)\s+(?:assigned|on|the)\s+(?:staff|crew|team)",
                    r"give\s+.*(?:information|info|detail).*\b(?:crew|staff)\b",
                    r"(?:show|tell)\s+.*\b(?:crew|staff)\b",
                ]
                is_crew_override = intent not in ("Data_Query", "Delay_Management") and any(re.search(p, prompt, re.IGNORECASE) for p in crew_override_patterns)
                if is_crew_override:
                    fids_override = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", prompt, re.IGNORECASE)
                    if not fids_override:
                        for msg in reversed(st.session_state.messages):
                            found = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", msg.get("content", ""), re.IGNORECASE)
                            if found:
                                fids_override = [f.upper().replace("_", "-") for f in found[:1]]
                                break
                    decision["intent"] = "Data_Query"
                    decision["extraction"]["flight_ids"] = fids_override
                    intent = "Data_Query"

                replace_match = re.search(r"find\s+(?:new\s+)?(?:replacement|replacements|crew)\s+(?:for|of)\s+([A-Z]{2,3}[-_]?\d{2,4})", prompt, re.IGNORECASE)
                if replace_match:
                    target_fid = replace_match.group(1).upper().replace("_", "-")
                    result = find_replacement_crew(target_fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH, use_llm=True)
                    if result["status"] == "success":
                        info = result.get("flight_info", {})
                        msg = f"### Crew Eligibility for {target_fid}\n"
                        msg += f"**Route:** {info.get('origin', '?')} → {info.get('destination', '?')} | "
                        msg += f"**Aircraft:** {info.get('aircraft_type', '?')} | "
                        msg += f"**Duration:** {info.get('flight_hours', 0)}h\n\n"

                        assigned = result.get("assigned_crew", [])
                        if assigned:
                            msg += "#### Currently Assigned Crew\n"
                            for c in assigned:
                                icon = "✅" if c["eligible"] else "❌"
                                viols = "; ".join(c.get("violations", [])) or "None"
                                msg += f"- {icon} **{c['name']}** ({c['role']}) - {c['status_label']}: {viols}\n"
                            msg += "\n"

                        eligible = result.get("eligible_standby", [])
                        if eligible:
                            msg += "#### Eligible Standby Crew (Suggested Replacements)\n"
                            for c in eligible:
                                quals = ", ".join(c.get("qualifications", []))
                                msg += f"- ✅ **{c['name']}** ({c['role']}) - Duty: {c['current_duty_hours']}h, Rolling: {c['rolling_7_day_hours']}h, Cost: ${c['cost']:.2f}, Quals: {quals}\n"
                            msg += "\n"
                        else:
                            msg += "#### Eligible Standby Crew\nNo standby crew are currently eligible.\n\n"

                        ineligible = result.get("ineligible", [])
                        if ineligible:
                            msg += "#### Ineligible Crew\n"
                            for c in ineligible:
                                reasons = "; ".join(c.get("violations", []))
                                msg += f"- ❌ **{c['name']}** ({c['role']}) [{c['status_label']}]: {reasons}\n"
                            msg += "\n"

                        summary = result.get("summary", {})
                        msg += f"---\n**Summary:** {summary.get('total_crew', 0)} total | "
                        msg += f"{summary.get('eligible_standby_count', 0)} eligible standby | "
                        msg += f"{summary.get('ineligible_count', 0)} ineligible"

                        st.markdown("### Replacement Crew")
                        st.write(msg)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "flights",
                            "content": msg, "decision": decision,
                        })
                        st.stop()
                    else:
                        st.error(result.get("message", "Could not find replacements."))
                        st.stop()

                if intent == "General_Chat":
                    result_text = _groq_general_chat(prompt)
                    st.markdown("### AI Assistant")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "chat",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Rule_Query":
                    legal_text = retrieve_legal_guidance(prompt)
                    category = decision.get("extraction", {}).get("category")
                    st.markdown("### Rule Analysis")
                    st.info(legal_text)
                    if category:
                        st.caption(f"Category: {category}")
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "rule",
                        "content": legal_text, "decision": decision,
                        "category": category,
                    })

                elif intent == "Flight_Status":
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    if flight_ids:
                        result_text = answer_flight_query(f"status of flights {' '.join(flight_ids)}")
                    else:
                        result_text = _groq_rag_chat(prompt)
                    st.markdown("### Flight Information")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "flights",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Data_Query":
                    extraction = decision.get("extraction", {})
                    extracted_fids = extraction.get("flight_ids", [])
                    prompt_fids = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", prompt, re.IGNORECASE)
                    all_fids = extracted_fids or [f.upper().replace("_", "-") for f in prompt_fids]

                    crew_keywords = ("crew", "staff", "who", "can fly", "eligible", "available", "roster", "information about")
                    has_crew_keyword = any(kw in prompt.lower() for kw in crew_keywords)

                    if not all_fids and has_crew_keyword:
                        for msg in reversed(st.session_state.messages):
                            content = msg.get("content", "")
                            found = re.findall(r"([A-Z]{2,3}[-_]?\d{2,4})", content, re.IGNORECASE)
                            if found:
                                all_fids = [f.upper().replace("_", "-") for f in found[:1]]
                                break

                    assigned_keywords = ("assigned", "on board", "onboard", "on this flight", "flying", "crew for", "crew of", "crew on", "its crew", "the crew")
                    is_assigned_query = any(kw in prompt.lower() for kw in assigned_keywords)

                    if all_fids and has_crew_keyword and is_assigned_query:
                        fid = all_fids[0]
                        flight = get_flight(fid, DEFAULT_DB_PATH)
                        if not flight:
                            st.error(f"Flight {fid} not found.")
                            st.stop()
                        assigned = get_crew_for_flight(fid, DEFAULT_DB_PATH)
                        msg = f"### Crew — {fid} ({flight.origin} → {flight.destination})\n\n"

                        if assigned:
                            crew_list = load_crew(str(DEFAULT_CSV_PATH))
                            crew_map = {c.crew_id: c for c in crew_list}
                            msg += "| Name | Role | Duty | Rolling 7d | Rest |\n"
                            msg += "|------|------|------|-----------|------|\n"
                            for a in assigned:
                                member = crew_map.get(a["crew_id"])
                                if member:
                                    msg += f"| {member.name} | {a['role']} | {member.current_duty_hours}h | {member.rolling_7_day_hours}h | {member.rest_status} |\n"
                                else:
                                    msg += f"| {a['crew_id']} | {a['role']} | - | - | - |\n"
                        else:
                            msg += "No crew assigned to this flight.\n"

                        st.markdown(msg)
                        with st.expander("System Extraction & Audit Trail"):
                            st.json(decision)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "data",
                            "content": msg, "decision": decision,
                        })

                    elif all_fids and has_crew_keyword:
                        fid = all_fids[0]
                        roster_scan = find_replacement_crew(fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH, use_llm=True)
                        if roster_scan["status"] == "success":
                            info = roster_scan.get("flight_info", {})
                            msg = f"### Crew Eligibility for {fid}\n"
                            msg += f"**Route:** {info.get('origin', '?')} → {info.get('destination', '?')} | "
                            msg += f"**Aircraft:** {info.get('aircraft_type', '?')} | "
                            msg += f"**Duration:** {info.get('flight_hours', 0)}h | "
                            msg += f"**Night Duty:** {'Yes' if info.get('is_night_duty') else 'No'}\n\n"

                            assigned = roster_scan.get("assigned_crew", [])
                            if assigned:
                                msg += "#### Currently Assigned Crew\n"
                                msg += "| Name | Role | Duty | Status | Violations |\n"
                                msg += "|------|------|------|--------|------------|\n"
                                for c in assigned:
                                    icon = "✅" if c["eligible"] else "❌"
                                    viols = "; ".join(c.get("violations", [])) or "None"
                                    msg += f"| {icon} {c['name']} | {c['role']} | {c['current_duty_hours']}h | {c['status_label']} | {viols} |\n"
                                msg += "\n"

                            eligible = roster_scan.get("eligible_standby", [])
                            if eligible:
                                msg += "#### Eligible Standby Crew (Suggested Additions)\n"
                                msg += "| Name | Role | Duty | Rolling 7d | Rest | Cost | Qualifications |\n"
                                msg += "|------|------|------|-----------|------|------|----------------|\n"
                                for c in eligible:
                                    quals = ", ".join(c.get("qualifications", []))
                                    msg += f"| ✅ {c['name']} | {c['role']} | {c['current_duty_hours']}h | {c['rolling_7_day_hours']}h | {c['rest_status']} | ${c['cost']:.2f} | {quals} |\n"
                                msg += "\n"
                            else:
                                msg += "#### Eligible Standby Crew\nNo standby crew are currently eligible for this flight.\n\n"

                            busy_eligible = roster_scan.get("eligible_assigned_elsewhere", [])
                            if busy_eligible:
                                msg += "#### Eligible Crew (Assigned Elsewhere)\n"
                                for c in busy_eligible:
                                    msg += f"- **{c['name']}** ({c['role']}) - Duty: {c['current_duty_hours']}h, Rolling: {c['rolling_7_day_hours']}h\n"
                                msg += "\n"

                            ineligible = roster_scan.get("ineligible", [])
                            if ineligible:
                                msg += "#### Ineligible Crew (DGCA Violations)\n"
                                for c in ineligible:
                                    reasons = "; ".join(c.get("violations", []))
                                    msg += f"- ❌ **{c['name']}** ({c['role']}) [{c['status_label']}]: {reasons}\n"
                                msg += "\n"

                            summary = roster_scan.get("summary", {})
                            msg += f"---\n**Summary:** {summary.get('total_crew', 0)} total | "
                            msg += f"{summary.get('eligible_standby_count', 0)} eligible standby | "
                            msg += f"{summary.get('assigned_to_flight', 0)} assigned | "
                            msg += f"{summary.get('ineligible_count', 0)} ineligible"

                            st.markdown("### Crew Roster Analysis")
                            st.write(msg)
                            with st.expander("System Extraction & Audit Trail"):
                                st.json(decision)
                            st.session_state.messages.append({
                                "role": "assistant", "type": "data",
                                "content": msg, "decision": decision,
                            })
                        else:
                            st.error(roster_scan.get("message", "Could not scan roster."))
                    else:
                        result_text = _groq_rag_chat(prompt)
                        st.markdown("### AI Assistant")
                        st.write(result_text)
                        with st.expander("System Extraction & Audit Trail"):
                            st.json(decision)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "data",
                            "content": result_text, "decision": decision,
                        })

                elif intent == "Compliance_Check":
                    extraction = decision.get("extraction", {})
                    crew_id = extraction.get("crew_id")
                    if crew_id:
                        result = validate_single_crew(crew_id, csv_path=str(DEFAULT_CSV_PATH))
                        eligible = result.get("compliance", {}).get("eligible", False)
                        violations = result.get("compliance", {}).get("violations", [])
                        warnings = result.get("compliance", {}).get("warnings", [])
                        lines = [f"**{result.get('name', crew_id)}** ({result.get('role', 'N/A')})"]
                        lines.append(f"Rest status: {result.get('rest_status', 'N/A')}")
                        lines.append(f"Duty hours: {result.get('current_duty_hours', 0)} | Rolling 7-day: {result.get('rolling_7_day_hours', 0)}")
                        if eligible:
                            lines.append("**ELIGIBLE** for assignment")
                        else:
                            lines.append("**NOT ELIGIBLE:**")
                            for v in violations:
                                lines.append(f"  - {v}")
                        if warnings:
                            lines.append("**Warnings:**")
                            for w in warnings:
                                lines.append(f"  - {w}")
                        result_text = "\n".join(lines)
                    else:
                        result_text = "Please specify a crew ID to check (e.g., 'check crew CRW001')."

                    st.markdown("### Compliance Check")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "compliance",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Schedule_Disruption":
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])

                    if flight_ids:
                        solver_result = solve_multi_flight(
                            flight_ids,
                            str(DEFAULT_CSV_PATH),
                            required_counts=extraction.get("required_counts"),
                        )
                        flight_count = len(flight_ids)
                        total_hours = solver_result.get("total_flight_hours", 0)
                        status = solver_result.get("status", "Unknown")
                        total_cost = solver_result.get("objective_value", 0.0)
                        selected_count = solver_result.get("selected_count", 0)

                        msg_text = f"**Recovery plan for {flight_count} flight(s)** ({total_hours}h total)\n"
                        if status == "Optimal":
                            msg_text += "Most cost-effective crew assignment found."
                        else:
                            msg_text += f"Warning: {status} - closest partial assignment."

                        st.markdown("### Multi-Flight Recovery Plan")
                        st.write(msg_text)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Cost", f"${total_cost:,.2f}")
                        col2.metric("Crew Selected", selected_count)
                        col3.metric("Flights Covered", flight_count)

                        df = pd.DataFrame(solver_result.get("selected_crew", []))
                        if not df.empty:
                            display_cols = [c for c in df.columns if not c.startswith("qualifications")]
                            st.dataframe(df[display_cols], hide_index=True)
                            st.download_button(
                                "Download Assignment (CSV)",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="crew_assignment.csv",
                                mime="text/csv",
                                key=f"dl_active_{len(st.session_state.messages)}",
                            )

                        st.session_state.messages.append({
                            "role": "assistant", "type": "multi_solver",
                            "content": msg_text, "decision": decision,
                            "total_cost": total_cost, "selected_count": selected_count,
                            "selected_crew": solver_result.get("selected_crew", []),
                            "flight_count": flight_count,
                        })
                    else:
                        hours = extraction.get("scenario_flight_hours", 3.0)
                        night = extraction.get("scenario_is_night_duty", True)
                        solver_result = solve_from_csv(
                            str(DEFAULT_CSV_PATH),
                            scenario_flight_hours=hours,
                            scenario_is_night_duty=night,
                            required_counts=extraction.get("required_counts"),
                        )
                        status = solver_result.get("status", "Unknown")
                        total_cost = solver_result.get("objective_value", 0.0)
                        selected_count = solver_result.get("selected_count", 0)

                        msg_text = "Action Plan Ready: Most cost-effective crew assignment:" if status == "Optimal" else f"Warning - {status}:"
                        st.markdown("### Operational Action Plan")
                        st.write(msg_text)
                        col1, col2 = st.columns(2)
                        col1.metric("Total Cost", f"${total_cost:,.2f}")
                        col2.metric("Crew Selected", selected_count)

                        df = pd.DataFrame(solver_result.get("selected_crew", []))
                        if not df.empty:
                            display_cols = [c for c in df.columns if not c.startswith("qualifications")]
                            st.dataframe(df[display_cols], hide_index=True)
                            st.download_button(
                                "Download Assignment (CSV)",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="crew_assignment.csv",
                                mime="text/csv",
                                key=f"dl_active_{len(st.session_state.messages)}",
                            )

                        st.session_state.messages.append({
                            "role": "assistant", "type": "solver",
                            "content": msg_text, "decision": decision,
                            "total_cost": total_cost, "selected_count": selected_count,
                            "selected_crew": solver_result.get("selected_crew", []),
                            "missing_roles": solver_result.get("missing_roles", {}),
                        })

                elif intent == "Delay_Management":
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    delay_minutes = extraction.get("delay_minutes")
                    is_cancel = extraction.get("is_cancel", False)

                    if flight_ids:
                        fid = flight_ids[0]
                        if is_cancel:
                            result = process_cancellation(fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                            if result["status"] == "success":
                                msg = f"**Flight {fid} Cancelled**\n\n"
                                if result["freed_crew"]:
                                    msg += f"**{result['freed_count']} crew freed** back to standby:\n"
                                    for c in result["freed_crew"]:
                                        msg += f"- {c['crew_id']} ({c['role']})\n"
                                else:
                                    msg += "No crew was assigned to this flight."
                            else:
                                msg = result.get("message", "Cancellation failed.")

                            st.markdown("### Flight Cancelled")
                            st.write(msg)
                            st.session_state.messages.append({
                                "role": "assistant", "type": "flights",
                                "content": msg, "decision": decision,
                            })

                        elif delay_minutes is not None and delay_minutes > 0:
                            impact = analyze_delay_impact(fid, delay_minutes, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                            roster_scan = find_replacement_crew(fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH, use_llm=True)

                            if impact["status"] == "success" and roster_scan["status"] == "success":
                                f_info = impact.get("flight_info", {})
                                msg = f"### Flight {fid} — Delay +{int(delay_minutes)} min\n"
                                msg += f"**Route:** {f_info.get('origin', '?')} → {f_info.get('destination', '?')} | "
                                msg += f"**Aircraft:** {f_info.get('aircraft_type', '?')} | "
                                msg += f"**Duration:** {f_info.get('flight_hours', 0)}h\n\n"

                                assigned = impact.get("assigned_crew", [])
                                ineligible_list = impact.get("ineligible_crew", [])

                                if assigned:
                                    ineligible_names = {c["crew_id"] for c in ineligible_list}
                                    msg += "**Assigned crew:**\n"
                                    for c in assigned:
                                        if c["crew_id"] in ineligible_names:
                                            viols = "; ".join(c.get("violations", []))
                                            msg += f"- ❌ **{c['name']}** ({c['role']}) — {viols}\n"
                                        else:
                                            msg += f"- ✅ **{c['name']}** ({c['role']})\n"
                                    msg += "\n"

                                eligible = roster_scan.get("eligible_standby", [])
                                if eligible:
                                    msg += "**Suggested replacements:**\n"
                                    for c in eligible:
                                        quals = ", ".join(c.get("qualifications", []))
                                        msg += f"- **{c['name']}** ({c['role']}) — Rest: {c['rest_status']}, Duty: {c['current_duty_hours']}h\n"
                                else:
                                    msg += "No eligible standby crew available.\n"

                                msg += f"\nProceed with this delay and assign replacements? (reply `yes` or `no`)"

                                st.markdown("### Delay Analysis")
                                st.write(msg)
                                st.session_state.pending_delay = {
                                    "flight_id": fid,
                                    "delay_minutes": delay_minutes,
                                }
                                st.session_state.messages.append({
                                    "role": "assistant", "type": "flights",
                                    "content": msg, "decision": decision,
                                })
                            else:
                                st.error(impact.get("message", "Delay analysis failed."))
                        else:
                            st.info(f"Tell me how much to delay {fid} by (e.g., 'Delay {fid} by 3 hours').")
                    else:
                        st.info("Please specify a flight ID to delay (e.g., 'Delay AI-501 by 2 hours').")

                elif intent == "Planning_Query":
                    extraction = decision.get("extraction", {})
                    result_text = _answer_planning_query(prompt, extraction)
                    st.markdown("### Flight Risk Forecast")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "planning",
                        "content": result_text, "decision": decision,
                    })

                else:
                    result_text = _groq_rag_chat(prompt)
                    st.markdown("### AI Assistant")
                    st.write(result_text)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "data",
                        "content": result_text, "decision": decision,
                    })

            except Exception as exc:
                st.error(f"Dispatcher failed: {exc}")

    components.html(
        """
        <script>
        (function () {
            var doc = parent.document;
            var key = "__airline_chat_count__";
            function chatPanelVisible() {
                var panels = doc.querySelectorAll('[data-testid="stTabPanel"]');
                if (!panels.length) return true;
                return panels[0].getClientRects().length > 0;
            }
            function syncBar() {
                var bar = doc.querySelector('[data-testid="stBottom"]');
                if (!bar) return;
                bar.style.display = chatPanelVisible() ? "" : "none";
            }
            syncBar();
            var msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
            var count = msgs.length;
            var prev = parent[key] || 0;
            if (count > prev && chatPanelVisible() && count) {
                msgs[count - 1].scrollIntoView({ behavior: "smooth", block: "end" });
            }
            parent[key] = count;
            var tabs = doc.querySelector('[data-testid="stTabs"]');
            if (tabs) {
                new MutationObserver(syncBar).observe(
                    tabs, { attributes: true, subtree: true, attributeFilter: ["aria-selected"] }
                );
            }
        })();
        </script>
        """,
        height=0,
    )


# ============================================================
# TAB 2: CREW
# ============================================================
with tab_ops:
    st.header("Crew")
    st.caption("Standby crew and flight-wise crew assignments.")

    if st.button("Sync Flights & Assign Crew", help="Re-insert OpenSky flights and auto-assign crew from standby roster."):
        with st.spinner("Syncing flights and assigning crew..."):
            sync_result = sync_opensky_flights_to_db(
                csv_path=str(DEFAULT_CSV_PATH), db_path=DEFAULT_DB_PATH
            )
        st.success(sync_result["message"])
        st.rerun()

    flights = get_flights(db_path=DEFAULT_DB_PATH)
    crew = load_crew(DEFAULT_CSV_PATH)

    if not flights:
        st.info("No flights in schedule. Click 'Sync Flights & Assign Crew' above to load OpenSky flights.")
    elif not crew:
        st.warning("No crew data found.")
    else:
        name_map = {m.crew_id: m.name for m in crew}

        all_assigned_ids = set()
        flight_crew_map = {}
        for f in flights:
            assigned = get_crew_for_flight(f.flight_id, DEFAULT_DB_PATH)
            flight_crew_map[f.flight_id] = assigned
            for a in assigned:
                all_assigned_ids.add(a["crew_id"])

        flight_labels = []
        _required_total = sum(REQUIRED_CREW.values())
        for f in flights:
            dep = f.std.strftime("%H:%M") if f.std else "?"
            n = len(flight_crew_map[f.flight_id])
            flight_labels.append("%s  %s->%s  dep %s  [%d/%d crew]" % (
                f.flight_id, f.origin, f.destination, dep, n, _required_total
            ))

        total_assigned = len(all_assigned_ids)
        total_crew = len(crew)
        total_standby = total_crew - total_assigned

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Flights", len(flights))
        m2.metric("Assigned Crew", total_assigned)
        m3.metric("Standby Crew", total_standby)

        st.divider()

        selected = st.selectbox("Select a flight", flight_labels)
        if st.button("Show Crew", type="primary"):
            sel_fid = selected.split("  ")[0]
            assigned = flight_crew_map[sel_fid]
            if assigned:
                crew_rows = []
                shift_lookup = _get_crew_shift_lookup()
                for a in assigned:
                    cid = a["crew_id"]
                    crew_rows.append({
                        "Crew ID": cid,
                        "Name": name_map.get(cid, "Unknown"),
                        "Job": a["role"],
                        "Shift": shift_lookup.get(cid, ""),
                        "Status": a.get("status", "assigned"),
                    })
                st.dataframe(pd.DataFrame(crew_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("No crew assigned to %s." % sel_fid)

        st.divider()

        st.subheader("Standby Crew")
        st.caption("Crew members not assigned to any flight today.")
        crew_data = [m.to_dict() for m in crew]
        df = pd.DataFrame(crew_data)
        standby_df = df[~df["crew_id"].isin(all_assigned_ids)].copy()
        if not standby_df.empty:
            display_df = pd.DataFrame({
                "Crew ID": standby_df["crew_id"],
                "Name": standby_df["name"],
                "Job": standby_df["role"],
                "Shift": standby_df.get("shift", ""),
                "Base": standby_df["base_airport"],
                "Rest Status": standby_df["rest_status"],
                "Duty Hours": standby_df["current_duty_hours"],
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("All crew members are assigned to flights.")


# ============================================================
# TAB 3: FORECASTING (Delay Prediction)
# ============================================================
with tab_forecast:
    st.header("Forecasting")
    st.caption("Predict delays and suggest DGCA-compliant crew assignments using OpenSky data and ML models.")

    st.subheader("Daily Briefing — BLR Flights")
    st.caption("One-click pipeline: seed latest data, predict delays, suggest DGCA-compliant crew assignments. Today's predictions weight this weekday's delay history from previous weeks (recency-decayed).")

    from data.opensky_db import get_flight_stats as get_opensky_stats
    _os_stats = get_opensky_stats(DEFAULT_DB_PATH)
    if _os_stats.get("total_flights", 0) > 0:
        _dr = _os_stats.get("date_range")
        if _dr and _dr[0] and _dr[1]:
            _d0 = datetime.strptime(_dr[0], "%Y-%m-%d").date()
            _d1 = datetime.strptime(_dr[1], "%Y-%m-%d").date()
            _n_days = (_d1 - _d0).days + 1
            st.info(
                f"**{_n_days}** days of data | "
                f"**{_os_stats['total_flights']}** flights | "
                f"**{_os_stats.get('weather_records', 0)}** weather records | "
                f"{_dr[0]} to {_dr[1]}"
            )
        else:
            st.info(f"**{_os_stats['total_flights']}** flights stored")

    col_seed, col_brief, col_retrain = st.columns(3)
    with col_seed:
        seed_days = st.slider("Days to seed", 1, 60, 30, key="seed_days",
                              help="Fetch historical flight data from OpenSky. Existing days are skipped automatically.")
        if st.button(f"Seed {seed_days} days of data", help="Seeds OpenSky data, recomputes delay labels and rotation chains."):
            with st.spinner(f"Fetching {seed_days} days of flight data..."):
                update_result = update_daily_data(days_back=seed_days, db_path=DEFAULT_DB_PATH)
            with st.spinner("Syncing flights and assigning crew..."):
                sync_result = sync_opensky_flights_to_db(
                    csv_path=str(DEFAULT_CSV_PATH), db_path=DEFAULT_DB_PATH
                )
            with st.spinner("Backfilling predictions for historical data..."):
                bf_count = backfill_predictions(DEFAULT_DB_PATH)
                act_count = update_actuals(DEFAULT_DB_PATH)
            seed_r = update_result["seed"]
            new_flights = seed_r["total_flights"]
            skipped = seed_r["days_skipped"]
            actual = seed_days - skipped
            st.session_state["system_initialized"] = True
            st.session_state["flight_count"] = seed_r["total_flights"]
            st.success(
                f"**{new_flights}** flights from **{actual}** new days "
                f"({skipped} days already stored). "
                f"Labels: {update_result['labels_added']}. "
                f"{sync_result['message']}. "
                f"Audit: {bf_count} predictions backfilled, {act_count} actuals updated."
            )
            st.rerun()
    with col_brief:
        if st.button("Run Daily Briefing", type="primary", help="Seeds yesterday's data, predicts delays, suggests crew."):
            with st.spinner("Running Daily Briefing — updating data, predicting delays, assigning crew..."):
                t0 = time.perf_counter()
                update_result = update_daily_data(days_back=1, db_path=DEFAULT_DB_PATH, retrain=False)
                t1 = time.perf_counter(); print(f"[perf] update_daily_data: {t1-t0:.2f}s")
                today_flights = get_today_schedule(db_path=DEFAULT_DB_PATH)
                t2 = time.perf_counter(); print(f"[perf] get_today_schedule: {t2-t1:.2f}s")
                crew_plan = proactive_crew_assignment(today_flights, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                t3 = time.perf_counter(); print(f"[perf] proactive_crew_assignment: {t3-t2:.2f}s")
                at_risk = get_at_risk_crew(str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                t4 = time.perf_counter(); print(f"[perf] get_at_risk_crew: {t4-t3:.2f}s")
                log_predictions(today_flights, DEFAULT_DB_PATH)
                t5 = time.perf_counter(); print(f"[perf] log_predictions: {t5-t4:.2f}s")
                update_actuals(DEFAULT_DB_PATH)
                t6 = time.perf_counter(); print(f"[perf] update_actuals: {t6-t5:.2f}s")
                print(f"[perf] TOTAL briefing: {t6-t0:.2f}s")
            st.session_state["today_flights"] = today_flights
            st.session_state["crew_plan"] = crew_plan
            st.session_state["at_risk_crew"] = at_risk
            st.rerun()

    with col_retrain:
        if st.button("Retrain Models", help="Retrain XGBoost delay prediction models with latest data."):
            with st.spinner("Retraining models..."):
                from ml_engine.delay_predictor import retrain_if_stale
                r = retrain_if_stale(max_age_hours=0, db_path=DEFAULT_DB_PATH)
            st.success(f"Models: {r.get('status', 'done')}")
            st.rerun()

    today_flights = st.session_state.get("today_flights", [])
    crew_plan = st.session_state.get("crew_plan", {})
    at_risk = st.session_state.get("at_risk_crew", {})
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if today_flights:
        high = [f for f in today_flights if f["prediction"]["risk_level"] == "High"]
        med = [f for f in today_flights if f["prediction"]["risk_level"] == "Medium"]
        low = [f for f in today_flights if f["prediction"]["risk_level"] == "Low"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Flights", len(today_flights))
        c2.metric("High Risk", len(high), delta=f"{len(high)} need backup crew", delta_color="inverse" if high else "off")
        c3.metric("Medium Risk", len(med), delta=f"{len(med)} on standby alert", delta_color="inverse" if med else "off")
        c4.metric("Low Risk", len(low))

        if crew_plan:
            summary = crew_plan.get("summary", {})
            st.caption(
                f"Staffing: {summary.get('high_risk_count', 0)} high-risk | "
                f"{summary.get('medium_risk_count', 0)} medium-risk | "
                f"{summary.get('low_risk_count', 0)} low-risk flights"
            )

        st.divider()

        recs = crew_plan.get("crew_recommendations", {})
        crew_name_df = pd.read_csv(DEFAULT_CSV_PATH)
        crew_name_map = dict(zip(crew_name_df["crew_id"], crew_name_df["name"]))
        risk_order = {"High": 0, "Medium": 1}
        high_med_flights = sorted(
            [f for f in today_flights if f["prediction"]["risk_level"] in ("High", "Medium")],
            key=lambda x: (risk_order.get(x["prediction"]["risk_level"], 2), -x["prediction"]["delay_probability"]),
        )
        for f in high_med_flights:
                pred = f["prediction"]
                risk_color = "red" if pred["risk_level"] == "High" else ("orange" if pred["risk_level"] == "Medium" else "gray")
                wx = f.get("weather", {})
                fid = f["callsign"]
                rec = recs.get(fid, {})
                suggestions = rec.get("suggested_crew", {})
                leg_type = f.get("leg_type", "First Leg")
                crew_action = f.get("crew_action", "")

                with st.container():
                    leg_badge = ":blue[First Leg]" if leg_type == "First Leg" else ":green[Return Leg]"
                    _dow_name = day_names[datetime.now().weekday()]
                    st.markdown(
                        f"**{fid}** | {f['scheduled_departure']} | "
                        f"**{f['route']}** | {f['avg_duration_min']}min | "
                        f":{risk_color}[**{pred['risk_level']}**] "
                        f"Delay prob: **{pred['delay_probability']*100:.0f}%** | "
                        f"Expected delay: **{pred['expected_delay_min']:.0f} min** | "
                        f"{leg_badge}"
                    )

                    if crew_action:
                        if leg_type == "Return Leg":
                            st.info(f"**{crew_action}**")
                        else:
                            st.warning(f"**{crew_action}**")

                    col_wx, col_hist = st.columns(2)
                    col_wx.caption(f"Weather: {wx.get('temp_c', '?')}C, Wind {wx.get('wind_kmh', '?')}km/h, Precip {wx.get('precipitation_mm', 0)}mm")
                    _hist_count = f.get("dow_sample_count") or f["total_flights"]
                    col_hist.caption(f"Weekday history ({_dow_name}): {f['delay_rate_pct']}% delayed, avg deviation {f['avg_deviation_min']}min across {_hist_count} {_dow_name}s")

                    if suggestions:
                        replacement_lines = []
                        for role_name, sug in suggestions.items():
                            old_cid = sug.get("old_crew_id")
                            old_name = sug.get("old_crew_name")
                            if not old_cid or not old_name:
                                current_crew = get_crew_for_flight(fid, DEFAULT_DB_PATH)
                                current_by_role = {}
                                for ac in current_crew:
                                    r = ac.get("role", "")
                                    if r not in current_by_role:
                                        current_by_role[r] = []
                                    current_by_role[r].append(ac)
                                holders = current_by_role.get(role_name, [])
                                old_cid = holders[0]["crew_id"] if holders else None
                                old_name = crew_name_map.get(old_cid, old_cid) if old_cid else "(unassigned)"
                            replacement_lines.append(f"**{role_name}**: {old_name} → **{sug['name']}** ({sug['crew_id']})")

                        if replacement_lines:
                            with st.expander(f"Replacement plan for **{fid}**", expanded=False):
                                for line in replacement_lines:
                                    st.markdown(f"• {line}")

                        n_roles = len(suggestions)
                        btn_label = f"Replace {n_roles} crew on {fid}" if n_roles > 1 else f"Replace crew on {fid}"
                        if st.button(btn_label, key=f"assign_{fid}"):
                            assigned_any = False
                            already_assigned_count = 0
                            failed_count = 0
                            reassigned_count = 0
                            replaced_count = 0
                            for role_name, sug in suggestions.items():
                                old_cid = sug.get("old_crew_id")
                                if old_cid:
                                    unassign_crew_from_flight(old_cid, fid, DEFAULT_DB_PATH)
                                    replaced_count += 1
                                role_count = sum(1 for a in get_crew_for_flight(fid, DEFAULT_DB_PATH) if a["role"] == role_name)
                                if role_count >= REQUIRED_CREW.get(role_name, 0):
                                    already_assigned_count += 1
                                    continue
                                old_flight = sug.get("assigned_flight", "")
                                if old_flight:
                                    unassign_crew_from_flight(sug["crew_id"], old_flight, DEFAULT_DB_PATH)
                                    reassigned_count += 1
                                ar = assign_crew_to_flight(sug["crew_id"], fid, role_name, DEFAULT_DB_PATH)
                                if ar.get("status") == "success":
                                    assigned_any = True
                                elif "already assigned" in ar.get("message", "").lower():
                                    already_assigned_count += 1
                                else:
                                    failed_count += 1
                            if assigned_any:
                                msg_lines = [f"**{fid}** crew updated:"]
                                for line in replacement_lines:
                                    msg_lines.append(f"• {line}")
                                msg = "\n".join(msg_lines)
                                extra = []
                                if replaced_count:
                                    extra.append(f"{replaced_count} old crew removed from {fid}")
                                if reassigned_count:
                                    extra.append(f"{reassigned_count} reassigned from other flights")
                                if extra:
                                    msg += "\n\n(" + ", ".join(extra) + ")"
                                st.success(msg)
                                st.session_state["crew_plan"] = proactive_crew_assignment(
                                    today_flights, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH
                                )
                                st.rerun()
                            elif already_assigned_count > 0:
                                st.warning(f"{already_assigned_count} crew already assigned to {fid}. Check if backup is actually needed.")
                            else:
                                st.error(f"Assignment failed for {failed_count} crew. They may have DGCA violations.")
                    elif fid in recs:
                        standby_count = rec.get("standby_count", 0)
                        if pred["risk_level"] in ("High", "Medium"):
                            if standby_count > 0:
                                st.success(f"Crew is DGCA-compliant. **{standby_count}** standby crew available as backup.")
                            else:
                                st.warning("No DGCA-compliant standby crew available for this flight.")
                        else:
                            st.info("Crew is DGCA-compliant. No changes needed.")

                    st.divider()

        st.subheader("All Flights")
        flights_data = []
        for f in today_flights:
            pred = f["prediction"]
            wx = f.get("weather", {})
            flights_data.append({
                "Flight": f["callsign"],
                "Time": f["scheduled_departure"],
                "Route": f["route"],
                "Duration": f"{f['avg_duration_min']}min",
                "Leg": f.get("leg_type", "First Leg"),
                "Weather": f"{wx.get('temp_c', '?')}C {wx.get('wind_kmh', '?')}km/h",
                "Risk": pred["risk_level"],
                "Delay Prob": f"{pred['delay_probability']*100:.0f}%",
                "Expected Delay": f"{pred['expected_delay_min']:.0f}min",
                "History": f"{f['delay_rate_pct']}% delayed ({f['total_flights']} flights)",
            })
        flights_df = pd.DataFrame(flights_data)
        st.dataframe(flights_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Delay Hotspots")

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            st.markdown("**By Departure Hour**")
            hourly = {}
            for f in today_flights:
                h = f["avg_departure_hour"]
                if h not in hourly:
                    hourly[h] = {"flights": 0, "delayed": 0}
                hourly[h]["flights"] += 1
                hourly[h]["delayed"] += f["delayed_count"]
            hourly_data = []
            for h in sorted(hourly.keys()):
                d = hourly[h]
                rate = d["delayed"] / d["flights"] * 100 if d["flights"] else 0
                hourly_data.append({"Hour": f"{h:02d}:00", "Flights": d["flights"], "Delayed": d["delayed"], "Rate": f"{rate:.0f}%"})
            st.dataframe(pd.DataFrame(hourly_data), hide_index=True, use_container_width=True)
        with col_h2:
            st.markdown("**By Route**")
            route_data = []
            seen_routes = {}
            for f in today_flights:
                r = f["route"]
                if r not in seen_routes:
                    seen_routes[r] = {"flights": 0, "delayed": 0, "max_dev": 0}
                seen_routes[r]["flights"] += 1
                seen_routes[r]["delayed"] += f["delayed_count"]
                seen_routes[r]["max_dev"] = max(seen_routes[r]["max_dev"], f["max_deviation_min"])
            for r, d in sorted(seen_routes.items(), key=lambda x: x[1]["delayed"], reverse=True):
                rate = d["delayed"] / d["flights"] * 100 if d["flights"] else 0
                route_data.append({"Route": r, "Flights": d["flights"], "Delayed": d["delayed"], "Rate": f"{rate:.0f}%", "Max Dev": f"{d['max_dev']:.0f}min"})
            st.dataframe(pd.DataFrame(route_data), hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Staffing Forecast (ML-Driven)")
        forecast = forecast_crew_needs(str(DEFAULT_CSV_PATH), today_schedule=today_flights)
        if "error" not in forecast:
            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Flights Today", forecast["flights_today"])
            fc2.metric("Expected Disruptions", forecast["expected_disruptions"])
            fc3.metric("Avg Flight Hours", f"{forecast['avg_flight_hours']:.1f}h")

            for role, data in forecast.get("role_breakdown", {}).items():
                status_color = "normal" if data["status"] == "Sufficient" else "inverse"
                metric_label = f"{role} ({data['status']})"
                metric_value = f"Available: {data['available']}"
                if data["gap"] > 0:
                    st.metric(metric_label, metric_value, f"Gap: {data['gap']}", delta_color=status_color)
                else:
                    st.metric(metric_label, metric_value)

    else:
        st.info("Click **Initialize System** (first time) then **Run Daily Briefing** to load predictions and crew suggestions.")

    with st.expander("Model Audit — Prediction vs Actual", expanded=False):
        audit_days = st.slider("Audit period (days)", 7, 60, 14, key="audit_days")
        audit_df = get_prediction_audit(days=audit_days, db_path=DEFAULT_DB_PATH)
        if audit_df.empty:
            st.info("No audited predictions yet. Run **Seed** or **Daily Briefing** to start logging predictions.")
        else:
            metrics = compute_audit_metrics(audit_df)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Predictions", metrics["total"])
            m2.metric("Accuracy", f"{metrics['accuracy']}%")
            m3.metric("Precision", f"{metrics['precision']}%")
            m4.metric("Recall", f"{metrics['recall']}%")
            m5.metric("MAE", f"{metrics['mae']}min")

            brier_col, f1_col = st.columns(2)
            brier_col.metric("Brier Score", f"{metrics['brier_score']}", help="Lower is better (0 = perfect)")
            f1_col.metric("F1 Score", f"{metrics['f1']}%")

            if metrics.get("calibration"):
                st.markdown("**Calibration — Predicted Probability vs Actual Delay Rate**")
                cal_df = pd.DataFrame(metrics["calibration"])
                cal_display = cal_df.rename(columns={
                    "bucket": "Predicted Range",
                    "predicted_avg": "Avg Predicted %",
                    "actual_rate": "Actual Delay %",
                    "count": "Flights",
                })
                st.dataframe(cal_display, hide_index=True, use_container_width=True)
                st.bar_chart(cal_df.set_index("bucket")[["predicted_avg", "actual_rate"]])

            if metrics.get("risk_distribution"):
                st.markdown("**Risk Level Performance**")
                risk_df = pd.DataFrame(metrics["risk_distribution"])
                risk_display = risk_df.rename(columns={
                    "risk_level": "Risk",
                    "count": "Predictions",
                    "actual_delay_rate": "Actual Delay %",
                    "avg_predicted_prob": "Avg Predicted %",
                    "avg_actual_delay_min": "Avg Actual Delay",
                })
                st.dataframe(risk_display, hide_index=True, use_container_width=True)

            if metrics.get("predictions"):
                st.markdown("**Recent Predictions**")
                pred_df = pd.DataFrame(metrics["predictions"])
                pred_display = pred_df.rename(columns={
                    "date": "Date",
                    "callsign": "Flight",
                    "origin": "From",
                    "destination": "To",
                    "delay_probability": "Pred Prob",
                    "risk_level": "Risk",
                    "actual_delay_min": "Actual Delay",
                    "actual_is_delayed": "Delayed?",
                    "match": "Result",
                })
                pred_display["Pred Prob"] = pred_display["Pred Prob"].apply(lambda x: f"{x*100:.0f}%")
                pred_display["Actual Delay"] = pred_display["Actual Delay"].apply(lambda x: f"{x:.0f}min" if pd.notna(x) else "—")
                pred_display["Delayed?"] = pred_display["Delayed?"].apply(lambda x: "Yes" if x == 1 else "No")
                st.dataframe(pred_display, hide_index=True, use_container_width=True)


# ============================================================
# TAB 4: LIVE TRACKING
# ============================================================
with tab_live:
    st.header("Live Flight Tracking — BLR (VOBL)")

    if "last_poll" not in st.session_state:
        st.session_state.last_poll = 0

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Refresh Live Data", type="primary"):
            try:
                result = poll_live_data(db_path=DEFAULT_DB_PATH)
                st.success(
                    f"Updated: {result['live_aircraft']} aircraft tracked, "
                    f"{result['delay_events']} delays detected"
                )
                st.session_state.last_poll = time.time()
            except Exception as e:
                st.error(f"Poll failed: {e}")
    with c2:
        auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)

    st.divider()

    flights = get_model_flights_with_status(db_path=DEFAULT_DB_PATH)

    if flights:
        table_data = []
        for f in flights:
            delay = f.get("delay_minutes")
            if delay is not None:
                if delay > 15:
                    delay_str = f"+{delay}min"
                elif delay > 0:
                    delay_str = f"+{delay}min"
                elif delay < 0:
                    delay_str = f"{delay}min"
                else:
                    delay_str = "On time"
            else:
                delay_str = "—"

            ft = f.get("flight_type", "One-way")
            rc = f.get("return_callsign", "")
            type_str = f"Round-trip ({rc})" if ft == "Round-trip" and rc else ft

            table_data.append({
                "Flight": f["callsign"],
                "Route": f"{f['origin']} → {f['destination']}",
                "Type": type_str,
                "Scheduled": f.get("scheduled_departure", "?"),
                "Duration": f"{f.get('avg_duration_min', 0)}min",
                "Status": f["status"],
                "Delay": delay_str,
                "Notes": f.get("notes", ""),
            })

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        status_counts = {}
        for f in flights:
            s = f["status"]
            status_counts[s] = status_counts.get(s, 0) + 1

        cols = st.columns(len(status_counts) or 1)
        for i, (status, count) in enumerate(status_counts.items()):
            cols[i].metric(status, count)
    else:
        st.info("No model flights found. Initialize the system first.")

    if auto_refresh:
        if time.time() - st.session_state.last_poll >= 300:
            try:
                poll_live_data(db_path=DEFAULT_DB_PATH)
                st.session_state.last_poll = time.time()
            except Exception:
                pass
            st.rerun()
        else:
            remaining = 300 - int(time.time() - st.session_state.last_poll)
            st.caption(f"Auto-refresh in {remaining}s")


# ============================================================
# TAB 5: PLANNING (Advance Crew Allocation)
# ============================================================
with tab_planning:
    from datetime import date as date_type

    def _render_actuals(selected_date):
        with st.spinner(f"Loading actuals for {selected_date}..."):
            actuals = get_flight_actuals_for_date(selected_date, db_path=DEFAULT_DB_PATH)
        st.divider()
        st.subheader(f"What Actually Happened — {selected_date}")
        st.caption("Times are UTC. Scheduled = typical recurring departure (as shown across the app); Actual = ADS-B first/last seen (includes taxi-out, so small gaps are normal). "
                   "Delayed = duration deviation beyond threshold, matching the rest of the app.")
        if not actuals:
            st.info("No flight data for this date.")
            return
        with_actual = [a for a in actuals if a["has_actual"]]
        delayed_rows = [a for a in with_actual if a["status"] == "Delayed"]
        pred_high = [a for a in with_actual if a.get("predicted_risk") == "High"]
        pred_med = [a for a in with_actual if a.get("predicted_risk") == "Medium"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flights Operated", len(with_actual))
        m2.metric("Delayed", len(delayed_rows), delta=f"{len(delayed_rows)/len(with_actual)*100:.0f}%" if with_actual else "0%",
                  delta_color="inverse" if delayed_rows else "off")
        m3.metric("Model High Risk", len(pred_high), delta_color="inverse" if pred_high else "off")
        m4.metric("Model Medium Risk", len(pred_med), delta_color="inverse" if pred_med else "off")

        actual_rows = []
        for a in actuals:
            row = {
                "Flight": a["callsign"],
                "Route": a["route"],
                "Scheduled Dep": a["scheduled_dep"] or "-",
                "Actual Dep": a["actual_dep"] or "-",
                "Actual Arr": a["actual_arr"] or "-",
                "Expected Time": f"{a['expected_flight_min']:.0f}min" if a["expected_flight_min"] is not None else "-",
                "Actual Time": f"{a['actual_flight_min']:.0f}min" if a["actual_flight_min"] is not None else "-",
                "Deviation": f"{a['deviation_min']:+.1f}min" if a["deviation_min"] is not None else "-",
                "Status": a["status"],
            }
            if a.get("predicted_prob") is not None:
                row["Predicted Prob"] = f"{a['predicted_prob']*100:.0f}%"
                row["Predicted Delay"] = f"{a['predicted_delay_min']:.0f}min" if a["predicted_delay_min"] is not None else "-"
                row["Actual Delay"] = f"{a['actual_delay_min']:+.1f}min" if a["actual_delay_min"] is not None else "-"
            actual_rows.append(row)

        actual_df = pd.DataFrame(actual_rows)
        styled = actual_df.style.map(
            lambda v: "color: #d63031; font-weight: bold" if v == "Delayed"
            else "color: #00b894; font-weight: bold" if v == "On time"
            else "",
            subset=["Status"],
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        if not with_actual:
            st.info("No ADS-B/label data found for the scheduled flights on this date.")

    st.header("Planning — Advance Crew Allocation")
    now_utc = datetime.utcnow()
    st.caption(f"Current date/time: {now_utc.strftime('%a %Y-%m-%d %H:%M')} UTC")

    today = date_type.today()
    max_date = today + timedelta(days=16)

    data_lo, _ = get_data_date_range(DEFAULT_DB_PATH)
    min_date = today
    if data_lo:
        try:
            data_start = date_type.fromisoformat(data_lo)
            if data_start < min_date:
                min_date = data_start
        except ValueError:
            pass

    selected_date = st.date_input(
        "Select a date to plan for",
        value=today,
        min_value=min_date,
        max_value=max_date,
    )

    if selected_date:
        days_ahead = (selected_date - today).days
        is_past = days_ahead < 0
        if days_ahead == 0:
            st.info("This is today. Use the **Forecasting** tab for the full Daily Briefing with crew assignment.")
        elif is_past:
            st.caption(f"Past date — showing **what actually happened** on {selected_date}.")
        else:
            st.caption(f"Planning {days_ahead} day(s) ahead — weather forecast available via Open-Meteo.")

        if is_past:
            _render_actuals(selected_date)
        else:
            with st.spinner(f"Predicting delays for {selected_date}..."):
                try:
                    planned_flights = get_schedule_for_date(selected_date, db_path=DEFAULT_DB_PATH)
                except Exception as e:
                    st.error(f"Failed to generate schedule: {e}")
                    planned_flights = []

        if not is_past and planned_flights:
            high = [f for f in planned_flights if f["prediction"]["risk_level"] == "High"]
            med = [f for f in planned_flights if f["prediction"]["risk_level"] == "Medium"]
            low = [f for f in planned_flights if f["prediction"]["risk_level"] == "Low"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Flights", len(planned_flights))
            c2.metric("High Risk", len(high), delta=f"{len(high)} need backup crew", delta_color="inverse" if high else "off")
            c3.metric("Medium Risk", len(med), delta=f"{len(med)} on standby alert", delta_color="inverse" if med else "off")
            c4.metric("Low Risk", len(low))

            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            st.caption(f"Day of week: **{day_names[selected_date.weekday()]}** — predictions weight this day's delay history from previous weeks (recency-decayed, newer weeks count more)")

            if high or med:
                st.divider()
                st.subheader("Crew Required by Role")

                high_count = len(high)
                med_count = len(med)
                total_high_med = high_count + med_count

                crew_breakdown = []
                for role_name, per_flight in REQUIRED_CREW.items():
                    high_needed = high_count * per_flight
                    med_needed = med_count * per_flight
                    total_needed = high_needed + med_needed
                    category = "Pilots" if role_name in ("Captain", "FO") else \
                               "Cabin Crew" if role_name == "CabinCrew" else \
                               "Ground — Ramp" if role_name == "RampAgent" else \
                               "Ground — Baggage" if role_name == "BaggageHandler" else \
                               "Ground — Cleaning" if role_name == "CabinCleaner" else \
                               "Ground — Check-in" if role_name == "CheckinAgent" else \
                               "Ground — Security"
                    crew_breakdown.append({
                        "Category": category,
                        "Role": role_name,
                        "Per Flight": per_flight,
                        "High Risk (x{})".format(high_count): high_needed,
                        "Medium Risk (x{})".format(med_count): med_needed,
                        "Total Needed": total_needed,
                    })

                st.dataframe(pd.DataFrame(crew_breakdown), use_container_width=True, hide_index=True)

                total_all = sum(r["Total Needed"] for r in crew_breakdown)
                st.metric("Total Standby Crew Needed", total_all,
                          help="Sum of all roles across High and Medium risk flights.")

                st.divider()
                st.subheader("Ground Staff Required by Shift")

                _GROUND_ROLES_LIST = ["RampAgent", "BaggageHandler", "CabinCleaner", "CheckinAgent", "SecurityAgent"]
                _ground_per_flight = sum(REQUIRED_CREW.get(r, 0) for r in _GROUND_ROLES_LIST)

                _flights_by_shift = {"Morning": 0, "Evening": 0, "Night": 0}
                for f in high + med:
                    dep_h = f.get("avg_departure_hour", 12)
                    if 6 <= dep_h < 14:
                        _flights_by_shift["Morning"] += 1
                    elif 14 <= dep_h < 22:
                        _flights_by_shift["Evening"] += 1
                    else:
                        _flights_by_shift["Night"] += 1

                _shift_windows_plan = {"Morning": "06:00-14:00", "Evening": "14:00-22:00", "Night": "22:00-06:00"}
                _all_crew_plan = load_crew(str(DEFAULT_CSV_PATH))
                _avail_by_shift = {"Morning": 0, "Evening": 0, "Night": 0}
                for _m in _all_crew_plan:
                    if _m.role in GROUND_ROLES and _m.shift in _avail_by_shift:
                        _avail_by_shift[_m.shift] += 1

                _shift_summary = []
                for _sn in ["Morning", "Evening", "Night"]:
                    _n_fl = _flights_by_shift[_sn]
                    _needed = _n_fl * _ground_per_flight
                    _avail = _avail_by_shift[_sn]
                    _shift_summary.append({
                        "Shift": f"{_sn} ({_shift_windows_plan[_sn]})",
                        "Flights": _n_fl,
                        "Ground Staff Needed": _needed,
                        "Ground Staff Available": _avail,
                        "Gap": _avail - _needed,
                    })

                st.dataframe(pd.DataFrame(_shift_summary), use_container_width=True, hide_index=True)

                _total_ground_needed = sum(r["Ground Staff Needed"] for r in _shift_summary)
                _total_ground_avail = sum(r["Ground Staff Available"] for r in _shift_summary)
                _gap_color = "normal" if _total_ground_avail >= _total_ground_needed else "inverse"
                st.metric("Ground Staff Total", f"{_total_ground_avail} available / {_total_ground_needed} needed",
                          delta=f"{_total_ground_avail - _total_ground_needed} surplus" if _total_ground_avail >= _total_ground_needed else f"{_total_ground_needed - _total_ground_avail} shortfall",
                          delta_color=_gap_color)

                st.divider()
                st.subheader("Ground Staff Shift Availability")

                _SHIFT_WINDOWS = {"Morning": (6, 14), "Evening": (14, 22), "Night": (22, 6)}
                all_crew = load_crew(str(DEFAULT_CSV_PATH))
                ground_by_role_shift = {}
                for member in all_crew:
                    if member.role in GROUND_ROLES:
                        role_val = member.role.value
                        shift = member.shift or "Unassigned"
                        ground_by_role_shift.setdefault(role_val, {}).setdefault(shift, 0)
                        ground_by_role_shift[role_val][shift] += 1

                ground_roles_list = ["RampAgent", "BaggageHandler", "CabinCleaner", "CheckinAgent", "SecurityAgent"]
                shift_names = ["Morning", "Evening", "Night", "Unassigned"]
                shift_grid = []
                for role_val in ground_roles_list:
                    row = {"Role": role_val}
                    for sn in shift_names:
                        row[sn] = ground_by_role_shift.get(role_val, {}).get(sn, 0)
                    shift_grid.append(row)
                st.dataframe(pd.DataFrame(shift_grid), use_container_width=True, hide_index=True)

                shift_desc = " | ".join(f"**{s}** {h[0]:02d}:00-{h[1]:02d}:00" for s, h in _SHIFT_WINDOWS.items())
                st.caption(f"Shift windows (UTC): {shift_desc}")

                st.divider()
                st.subheader("Flight-Shift Mapping")

                def _get_shift_name(dep_hour):
                    if 6 <= dep_hour < 14:
                        return "Morning"
                    elif 14 <= dep_hour < 22:
                        return "Evening"
                    else:
                        return "Night"

                fs_data = []
                for f in high + med:
                    dep_h = f.get("avg_departure_hour", 12)
                    dep_m = f.get("avg_departure_minute", 0)
                    shift_name = _get_shift_name(dep_h)
                    pred = f["prediction"]
                    fs_data.append({
                        "Flight": f["callsign"],
                        "Route": f["route"],
                        "Departure": f"{dep_h:02d}:{dep_m:02d}",
                        "Shift": shift_name,
                        "Risk": pred["risk_level"],
                        "Delay Prob": f"{pred['delay_probability']*100:.0f}%",
                        "Factors": "; ".join(pred.get("factors", [])[:2]),
                    })
                if fs_data:
                    fs_df = pd.DataFrame(fs_data)
                    st.dataframe(fs_df, use_container_width=True, hide_index=True)

                    for f in high + med:
                        dep_h = f.get("avg_departure_hour", 12)
                        shift_name = _get_shift_name(dep_h)
                        avail_ground = sum(
                            ground_by_role_shift.get(r, {}).get(shift_name, 0)
                            for r in ground_roles_list
                        )
                        needed_ground = sum(REQUIRED_CREW.get(r, 0) for r in ground_roles_list)
                        fid = f["callsign"]
                        risk = f["prediction"]["risk_level"]
                        if avail_ground < needed_ground:
                            st.warning(f"{fid} ({risk}) — {shift_name} shift: only {avail_ground} ground staff available, need {needed_ground}")
                        else:
                            st.success(f"{fid} ({risk}) — {shift_name} shift: {avail_ground} ground staff available, {needed_ground} needed")

            st.divider()
            st.subheader("All Flights")
            all_data = []
            for f in planned_flights:
                pred = f["prediction"]
                dep_h = f.get("avg_departure_hour", 12)
                all_data.append({
                    "Flight": f["callsign"],
                    "Route": f["route"],
                    "Time": f["scheduled_departure"],
                    "Duration": f"{f['avg_duration_min']}min",
                    "Leg": f.get("leg_type", "First Leg"),
                    "Shift": _get_shift_name(dep_h),
                    "Risk": pred["risk_level"],
                    "Delay Prob": f"{pred['delay_probability']*100:.0f}%",
                    "Expected Delay": f"{pred['expected_delay_min']:.0f}min",
                "History": f"{f['delay_rate_pct']}% on {day_names[datetime.now().weekday()]}s ({f.get('dow_sample_count') or f['total_flights']} flights)",
                })
            st.dataframe(pd.DataFrame(all_data), use_container_width=True, hide_index=True)
        elif not is_past:
            st.warning("No flights found for this date.")
