from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine import retrieve_legal_guidance
from router import route_request
from solver import solve_from_csv, solve_multi_flight
from data.flights_db import (
    init_db, get_flights, get_flight,
    get_disrupted_flights, get_upcoming_flights,
    get_flight_stats, get_crew_for_flight,
    assign_crew_to_flight,
)
from data.crew_loader import load_crew
from data.models import FlightStatus
from agents.flight_agent import answer_flight_query, answer_crew_query
from agents.crew_agent import find_eligible_crew_for_flight
from agents.compliance_agent import validate_single_crew, batch_validate
from ml_engine.resource_augmenter import forecast_crew_needs
from data.delay_handler import process_delay, process_cancellation, find_replacement_crew
from data.delay_handler import analyze_delay_impact, proactive_crew_assignment
from data.opensky_db import (
    poll_live_data, get_today_schedule, update_daily_data,
    get_model_flights_with_status, sync_opensky_flights_to_db,
)

APP_TITLE = "Airline Crew Operations Hub"
DEFAULT_CSV_PATH = Path(__file__).parent / "crew_standby_list.csv"
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "flights.db"

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
        from data.opensky_db import get_flight_schedule
        schedule = get_flight_schedule(limit=20, db_path=DEFAULT_DB_PATH)
        if schedule:
            lines.append("TODAY'S BLR FLIGHTS (20):")
            for f in schedule:
                lines.append(
                    "  %s %s->%s dep %02d:%02d %dmin delay_rate=%d%% risk_weight=%.1f" % (
                        f["callsign"], f["origin"], f["destination"],
                        f["avg_departure_hour"], f["avg_departure_minute"],
                        f["avg_duration_min"], f["delay_rate_pct"],
                        f["delay_rate_pct"] * f["avg_deviation_min"] / 100,
                    )
                )
    except Exception:
        pass

    try:
        from data.crew_loader import load_crew
        crew = load_crew(str(DEFAULT_CSV_PATH))
        if crew:
            lines.append("")
            lines.append("STANDBY CREW (%d members):" % len(crew))
            for c in crew:
                lines.append(
                    "  %s %s %s duty=%.1fh rolling=%.1fh rest=%s quals=%s" % (
                        c.crew_id, c.name, c.role.value,
                        c.current_duty_hours, c.rolling_7_day_hours,
                        c.rest_status,
                        ", ".join(q.aircraft_type for q in c.qualifications) if c.qualifications else "None",
                    )
                )
    except Exception:
        pass

    try:
        from data.flights_db import get_flights, get_crew_for_flight
        flights = get_flights(db_path=DEFAULT_DB_PATH)
        if flights:
            lines.append("")
            lines.append("CREW ASSIGNMENTS:")
            for f in flights:
                assigned = get_crew_for_flight(f.flight_id, DEFAULT_DB_PATH)
                if assigned:
                    crew_names = ["%s(%s)" % (a["crew_id"], a["role"]) for a in assigned]
                    lines.append("  %s: %s" % (f.flight_id, ", ".join(crew_names)))
    except Exception:
        pass

    return "\n".join(lines)


def _groq_rag_chat(question: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL
    from rag_engine import retrieve_legal_guidance

    context = _build_chat_context()

    dgca_context = ""
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
        "When asked about delays, check the assigned crew's duty hours and rolling 7-day hours "
        "against DGCA limits (Captain/FO: 12h duty, 35h rolling; CabinCrew: 14h duty, 45h rolling; "
        "GroundStaff: 10h duty). Flag any violations and suggest replacements from the standby crew list.\n"
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


# === 4 MAIN TABS ===
tab_chat, tab_ops, tab_forecast, tab_live = st.tabs([
    "Chat", "Operations", "Forecasting", "Live Tracking",
])


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

    if prompt := st.chat_input("Ask about flights, crew, rules, or disruptions..."):
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

                if intent == "Rule_Query":
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


# ============================================================
# TAB 2: OPERATIONS (Flights + Crew + Disruptions)
# ============================================================
with tab_ops:
    st.header("Operations")
    st.caption("Manage flights, crew roster, and handle disruptions.")

    op_tab_flights, op_tab_crew, op_tab_disruptions = st.tabs([
        "Flights", "Crew", "Disruptions & Reports",
    ])

    # --- Operations > Flights ---
    with op_tab_flights:
        st.subheader("BLR Flight Schedule")
        st.caption("20 flights from OpenSky with crew assignments and DGCA compliance status.")

        if st.button("Sync Flights & Assign Crew", help="Re-insert OpenSky flights and auto-assign crew from standby roster."):
            with st.spinner("Syncing flights and assigning crew..."):
                sync_result = sync_opensky_flights_to_db(
                    csv_path=str(DEFAULT_CSV_PATH), db_path=DEFAULT_DB_PATH
                )
            st.success(sync_result["message"])
            st.rerun()

        flights = get_flights(db_path=DEFAULT_DB_PATH)
        if flights:
            for f in flights:
                assigned = get_crew_for_flight(f.flight_id, DEFAULT_DB_PATH)
                dep_time = f.std.strftime("%H:%M") if f.std else "?"
                status_icon = {"scheduled": "🟢", "delayed": "🟡", "cancelled": "🔴"}.get(f.status.value.lower(), "⚪")

                with st.expander("%s %s %s→%s dep %s %dmin %s" % (
                    status_icon, f.flight_id, f.origin, f.destination,
                    dep_time, f.flight_duration_min, f.status.value,
                )):
                    st.write("**Aircraft:** %s | **Pax:** %d | **International:** %s" % (
                        f.aircraft_type, f.pax_count, "Yes" if f.is_international else "No"
                    ))

                    if assigned:
                        st.write("**Assigned Crew:**")
                        crew_data = []
                        for a in assigned:
                            crew_data.append({
                                "ID": a["crew_id"],
                                "Role": a["role"],
                                "Status": a.get("status", "assigned"),
                            })
                        st.dataframe(pd.DataFrame(crew_data), hide_index=True)
                    else:
                        st.warning("No crew assigned to this flight.")

                    if f.disruption_reason:
                        st.error("**Disruption:** %s" % f.disruption_reason)
        else:
            st.info("No flights in schedule. Click 'Sync Flights & Assign Crew' above to load OpenSky flights.")

    # --- Operations > Crew ---
    with op_tab_crew:
        st.subheader("Crew Assignments")
        st.caption("Flight-wise crew dashboard for the 20 BLR flights.")

        flights = get_flights(db_path=DEFAULT_DB_PATH)
        crew = load_crew(DEFAULT_CSV_PATH)

        if not flights:
            st.info("No flights in schedule.")
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
            for f in flights:
                dep = f.std.strftime("%H:%M") if f.std else "?"
                n = len(flight_crew_map[f.flight_id])
                flight_labels.append("%s  %s->%s  dep %s  [%d/8 crew]" % (
                    f.flight_id, f.origin, f.destination, dep, n
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
                    for a in assigned:
                        crew_rows.append({
                            "Crew ID": a["crew_id"],
                            "Name": name_map.get(a["crew_id"], "Unknown"),
                            "Job": a["role"],
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
                    "Base": standby_df["base_airport"],
                    "Rest Status": standby_df["rest_status"],
                    "Duty Hours": standby_df["current_duty_hours"],
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("All crew members are assigned to flights.")

    # --- Operations > Disruptions ---
    with op_tab_disruptions:
        st.subheader("Disruptions & Reports")
        st.caption("Handle delayed flights, find replacement crew, and generate compliance reports.")

        st.markdown("#### Disruption Management")
        st.caption("Flights marked as delayed or cancelled. Click to find replacement crew.")
        disrupted = get_disrupted_flights(DEFAULT_DB_PATH)
        if disrupted:
            for f in disrupted:
                with st.expander(f"{f.flight_id} - {f.status.value} ({f.origin}->{f.destination})"):
                    st.write(f"**Reason:** {f.disruption_reason or 'N/A'}")
                    st.write(f"**Aircraft:** {f.aircraft_type} | **Std:** {f.std.strftime('%H:%M')} | **Pax:** {f.pax_count}")
                    assigned = get_crew_for_flight(f.flight_id, DEFAULT_DB_PATH)
                    if assigned:
                        st.write(f"**Assigned Crew ({len(assigned)}):**")
                        for a in assigned:
                            st.write(f"  - {a['crew_id']} ({a['role']})")
                    if st.button(f"Find replacement crew for {f.flight_id}", key=f"find_{f.flight_id}"):
                        result = find_eligible_crew_for_flight(f.flight_id, str(DEFAULT_CSV_PATH))
                        if "error" not in result:
                            st.write(f"**{result['eligible_count']} eligible crew:**")
                            for c in result["eligible_crew"][:8]:
                                st.write(f"  {c['name']} ({c['role']}) - ${c['cost']:.2f}")
                        else:
                            st.error(result["error"])
        else:
            st.info("No disrupted flights.")

        st.markdown("#### Upcoming Flights")
        st.caption("Flights departing in the next 12 hours.")
        upcoming = get_upcoming_flights(hours_ahead=12, db_path=DEFAULT_DB_PATH)
        if upcoming:
            for f in upcoming:
                st.write(f"**{f.flight_id}** {f.origin}->{f.destination} at {f.std.strftime('%H:%M')} ({f.aircraft_type}) - {f.status.value}")
        else:
            st.info("No upcoming flights in the next 12 hours.")

        st.markdown("#### Batch Compliance Report")
        st.caption("Check which crew members are eligible for assignment right now.")
        if st.button("Generate Compliance Report"):
            report = batch_validate(str(DEFAULT_CSV_PATH))
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Crew", report["total"])
            c2.metric("Eligible", report["eligible"])
            c3.metric("Ineligible", report["ineligible"])

            report_df = pd.DataFrame([
                {
                    "ID": k, "Name": v["name"], "Role": v["role"],
                    "Eligible": "Yes" if v["eligible"] else "No",
                    "Issues": "; ".join(v["violations"]) if v["violations"] else "None",
                    "Warnings": "; ".join(v["warnings"]) if v["warnings"] else "None",
                }
                for k, v in report["details"].items()
            ])
            st.dataframe(report_df, hide_index=True)

        st.markdown("#### Flight Statistics")
        st.caption("Overview of flights by status and aircraft type.")
        stats = get_flight_stats(DEFAULT_DB_PATH)
        if stats["total"] > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.json(stats["by_status"])
            with col2:
                st.json(stats["by_aircraft"])


# ============================================================
# TAB 3: FORECASTING (Delay Prediction)
# ============================================================
with tab_forecast:
    st.header("Forecasting")
    st.caption("Predict delays and suggest DGCA-compliant crew assignments using OpenSky data and ML models.")

    st.subheader("Daily Briefing — BLR Flights")
    st.caption("One-click pipeline: seed latest data, predict delays, suggest DGCA-compliant crew assignments.")

    col_init, col_brief = st.columns(2)
    with col_init:
        sys_ready = st.session_state.get("system_initialized", False)
        flight_cnt = st.session_state.get("flight_count", 0)
        if sys_ready:
            st.button(
                "Initialize System (one-time)",
                disabled=True,
                help=f"Already loaded — {flight_cnt} flights and ML model found on disk. No API calls needed.",
            )
        else:
            if st.button("Initialize System (one-time)", help="Seeds 7 days of historical OpenSky data and trains the XGBoost model. Run this first time only."):
                with st.spinner("Seeding 7 days of flight data (~3 min)..."):
                    update_result = update_daily_data(days_back=7, db_path=DEFAULT_DB_PATH)
                with st.spinner("Syncing flights and assigning crew..."):
                    sync_result = sync_opensky_flights_to_db(
                        csv_path=str(DEFAULT_CSV_PATH), db_path=DEFAULT_DB_PATH
                    )
                seed_r = update_result["seed"]
                st.session_state["system_initialized"] = True
                st.session_state["flight_count"] = seed_r["total_flights"]
                st.success(
                    f"Seeded {seed_r['total_flights']} flights, "
                    f"{seed_r['total_weather_records']} weather records. "
                    f"Labels: {update_result['labels_added']}. "
                    f"Model: {update_result['model'].get('status', 'unknown')}. "
                    f"{sync_result['message']}"
                )
                st.rerun()
    with col_brief:
        if st.button("Run Daily Briefing", type="primary", help="Seeds yesterday's data, retrains if stale, predicts delays, suggests crew."):
            with st.spinner("Updating data + predicting delays + finding crew..."):
                update_result = update_daily_data(days_back=1, db_path=DEFAULT_DB_PATH)
                today_flights = get_today_schedule(db_path=DEFAULT_DB_PATH)
                crew_plan = proactive_crew_assignment(today_flights, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
            st.session_state["today_flights"] = today_flights
            st.session_state["crew_plan"] = crew_plan
            st.rerun()

    today_flights = st.session_state.get("today_flights", [])
    crew_plan = st.session_state.get("crew_plan", {})

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

        high_med = [f for f in today_flights if f["prediction"]["risk_level"] in ("High", "Medium")]
        if high_med:
            st.subheader("Flights Requiring Attention")

            recs = crew_plan.get("crew_recommendations", {})
            for f in high_med:
                pred = f["prediction"]
                risk_color = "red" if pred["risk_level"] == "High" else "orange"
                wx = f.get("weather", {})
                fid = f["callsign"]
                rec = recs.get(fid, {})
                suggestions = rec.get("suggested_crew", {})

                with st.container():
                    st.markdown(
                        f"**{fid}** | {f['scheduled_departure']} | "
                        f"**{f['route']}** | {f['avg_duration_min']}min | "
                        f":{risk_color}[**{pred['risk_level']}**] "
                        f"Delay prob: **{pred['delay_probability']*100:.0f}%** | "
                        f"Expected delay: **{pred['expected_delay_min']:.0f} min**"
                    )

                    col_wx, col_hist = st.columns(2)
                    col_wx.caption(f"Weather: {wx.get('temp_c', '?')}C, Wind {wx.get('wind_kmh', '?')}km/h, Precip {wx.get('precipitation_mm', 0)}mm")
                    col_hist.caption(f"History: {f['delay_rate_pct']}% delay rate, avg deviation {f['avg_deviation_min']}min across {f['total_flights']} flights")

                    if suggestions:
                        st.markdown("**Suggested Backup Crew (DGCA-Compliant):**")
                        sug_cols = st.columns(len(suggestions))
                        for i, (role_name, sug) in enumerate(suggestions.items()):
                            with sug_cols[i]:
                                st.markdown(
                                    f"**{role_name}**: {sug['name']}\n"
                                    f"Rest: {sug['rest_status']} | Duty: {sug['duty_hours']}h | "
                                    f"Rolling 7d: {sug['rolling_7d']}h"
                                )
                        if st.button(f"Assign suggested crew to {fid}", key=f"assign_{fid}"):
                            assigned_any = False
                            for role_name, sug in suggestions.items():
                                ar = assign_crew_to_flight(sug["crew_id"], fid, role_name, DEFAULT_DB_PATH)
                                if ar.get("status") == "success":
                                    assigned_any = True
                            if assigned_any:
                                st.success(f"Crew assigned to {fid}")
                                st.rerun()
                            else:
                                st.error("Assignment failed — crew may have DGCA violations.")
                    elif rec.get("standby_count", 0) == 0:
                        st.warning("No eligible standby crew available for this flight.")
                    else:
                        st.info(f"{rec.get('standby_count', 0)} standby crew available but none matched required roles.")

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
