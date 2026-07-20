from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from rag_engine import retrieve_legal_guidance, DGCA_RULES_DIR
from router import route_request
from solver import solve_from_csv, solve_multi_flight
from data.flights_db import (
    init_db, get_flights, get_flight, insert_flight, insert_flights,
    update_flight_status, get_disrupted_flights, get_upcoming_flights,
    get_flight_stats, clear_db, get_crew_for_flight, get_all_assignments,
)
from data.crew_loader import load_crew
from data.models import Flight, FlightStatus, Role
from data.staff_manager import auto_assign_flight, create_staff, get_assignment_summary
from agents.flight_agent import query_flights, get_disruption_summary, answer_flight_query
from agents.crew_agent import find_eligible_crew_for_flight, answer_crew_for_flight_query
from agents.compliance_agent import validate_single_crew, batch_validate
from agents.flight_agent import get_crew_availability, answer_crew_query
from ml_engine.delay_predictor import predict_delay, get_delay_insights
from ml_engine.resource_augmenter import (
    score_crew_utilization, find_optimal_swaps,
    forecast_crew_needs, get_augmentation_report,
)
from ml_engine.demand_forecaster import forecast_demand, get_demand_summary
from data.delay_handler import process_delay, process_cancellation, find_replacement_crew
from data.ops_loader import load_ops_dataset, get_ops_summary, get_flight_full_profile, create_ops_tables
from data.ground_ops.staff_analytics import (
    get_staff_role_distribution, get_shift_coverage_analysis,
    identify_understaffed_periods, get_staff_utilization,
)
from data.ground_ops.flow_analytics import (
    get_passenger_profile, get_baggage_load_profile,
    get_demand_by_route, predict_baggage_load, get_connecting_pax_analysis,
)
from data.ground_ops.turnaround_analytics import (
    get_turnaround_profile, get_boarding_efficiency, get_maintenance_impact,
)
from data.ground_ops.security_analytics import (
    get_security_throughput, get_screening_staff_performance, predict_queue_buildup,
)
from data.ground_ops.retail_analytics import (
    get_revenue_by_flight, get_revenue_by_gate,
    get_passenger_spend_profile, predict_retail_demand,
)
from data.recovery_engine import (
    assess_disruption_impact, find_recovery_options, get_disruption_cascade,
)
from ml_engine.delay_predictor import (
    get_delay_cause_breakdown, get_delay_by_airport, get_delay_by_time,
    get_delay_by_route_type, invalidate_profiles_cache,
)

APP_TITLE = "Airline Crew Operations Hub"
DEFAULT_CSV_PATH = Path(__file__).parent / "crew_standby_list.csv"
DEFAULT_DB_PATH = Path(__file__).parent / "data" / "flights.db"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Flight scheduling, crew management, DGCA compliance, and disruption recovery — all in one place.")

with st.expander("Welcome — How to Use This App", expanded=st.session_state.get("show_welcome", True)):
    st.markdown("""
    ### Quick Start
    1. **Load Sample Data** — Click `Load Sample Flights` in the sidebar to populate test flights
    2. **Explore the Tabs** — Use the five tabs below to manage different aspects of operations
    3. **Ask the Chat** — Type natural language questions to get instant answers

    ### What You Can Do

    | Tab | Purpose | Example Use |
    |-----|---------|-------------|
    | **Chat (Ask Anything)** | Ask questions in plain English | "What crew are available for AI-501?" |
    | **Flights** | View and add flights | Add a new flight, filter by airport |
    | **Crew** | Check crew eligibility, create staff | Run batch compliance checks, add new crew |
    | **Disruptions & Reports** | Handle disruptions | Find crew for delayed flights |
    | **ML & Forecasting** | Predict delays, forecast demand | "What's the delay risk for DEL-BOM at 8am?" |

    ### Tips
    - Use the **sidebar** to load data and filter flights
    - Click **expanders** (▼) in responses to see the full audit trail
    - **Download buttons** let you export crew assignments as CSV
    - **ML & Forecasting** tab helps you make data-driven decisions
    - **New flights auto-assign crew** from the standby roster
    - **Create staff** in the Crew tab to expand your pool
    """)
    if st.button("Got it, hide this"):
        st.session_state.show_welcome = False
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_initialized" not in st.session_state:
    init_db(DEFAULT_DB_PATH)
    st.session_state.db_initialized = True


def seed_sample_flights():
    sample = [
        Flight("AI-301", "DEL", "BOM", datetime.now().replace(hour=6, minute=30), "B737", FlightStatus.SCHEDULED, "A12", "T1", 165, 45, 130, False),
        Flight("AI-302", "BOM", "DEL", datetime.now().replace(hour=9, minute=0), "B737", FlightStatus.SCHEDULED, "B08", "T2", 170, 45, 125, False),
        Flight("AI-501", "DEL", "CCU", datetime.now().replace(hour=10, minute=30), "A320", FlightStatus.DELAYED, "C05", "T1", 140, 40, 140, False, "Weather delay"),
        Flight("AI-502", "CCU", "DEL", datetime.now().replace(hour=14, minute=0), "A320", FlightStatus.SCHEDULED, "D03", "T1", 150, 40, 135, False),
        Flight("AI-701", "DEL", "BLR", datetime.now().replace(hour=16, minute=45), "A321", FlightStatus.SCHEDULED, "A08", "T1", 180, 50, 170, False),
        Flight("AI-702", "BLR", "DEL", datetime.now().replace(hour=20, minute=30), "A321", FlightStatus.SCHEDULED, "E02", "T2", 155, 50, 165, True),
        Flight("AI-901", "DEL", "BOM", datetime.now().replace(hour=23, minute=15), "B737", FlightStatus.SCHEDULED, "A01", "T1", 120, 45, 125, True),
        Flight("AI-101", "DEL", "MAA", datetime.now().replace(hour=7, minute=0), "B737", FlightStatus.SCHEDULED, "A15", "T1", 145, 45, 155, False),
        Flight("AI-102", "MAA", "DEL", datetime.now().replace(hour=11, minute=0), "B737", FlightStatus.DEPARTED, "C01", "T2", 160, 45, 150, False),
        Flight("AI-201", "DEL", "HYD", datetime.now().replace(hour=13, minute=30), "A320", FlightStatus.SCHEDULED, "B10", "T1", 135, 40, 115, False),
    ]
    insert_flights(sample, DEFAULT_DB_PATH)
    
    for flight in sample:
        auto_assign_flight(flight.flight_id, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)


with st.sidebar:
    st.header("Quick Setup")
    st.caption("Start here to load data into the app.")
    if st.button("Load Sample Flights"):
        seed_sample_flights()
        st.success("10 sample flights loaded.")
    if st.button("Clear Flight Data"):
        clear_db(DEFAULT_DB_PATH)
        st.info("Flight data cleared.")

    st.divider()
    st.subheader("Operations Dataset")
    st.caption("Load the airport-operations-dataset (flights, staff, passengers, baggage, security, maintenance, retail).")
    if st.button("Load Operations Dataset"):
        with st.spinner("Loading 8 CSV files..."):
            result = load_ops_dataset(DEFAULT_DB_PATH)
        if result["status"] == "success":
            st.success(f"Loaded: {result['flights']} flights, {result['staff']} staff, {result['shifts']} shifts, {result['passengers']} pax, {result['baggage']} bags, {result['security']} screenings, {result['maintenance']} maintenance, {result['retail']} retail")
        else:
            st.error("Failed to load dataset.")

    ops_summary = get_ops_summary(DEFAULT_DB_PATH)
    if ops_summary.get("loaded"):
        with st.expander("Dataset Summary", expanded=False):
            for table, count in ops_summary.items():
                if table != "loaded" and count > 0:
                    st.caption(f"{table}: {count} rows")

    stats = get_flight_stats(DEFAULT_DB_PATH)
    st.subheader("Flight Stats")
    st.metric("Total Flights", stats.get("total", 0))
    for status, count in stats.get("by_status", {}).items():
        st.metric(f"  {status.title()}", count)

    st.divider()
    st.subheader("Filter Flights")
    origin_filter = st.selectbox("Origin Airport", ["All", "DEL", "BOM", "CCU", "BLR", "MAA", "HYD"], index=0)
    hours_ahead = st.slider("Show flights within (hours)", 1, 24, 6)

    st.divider()
    st.subheader("Knowledge Base")
    st.caption("PDFs used for DGCA compliance rules.")
    kb_path = Path("knowledge_base")
    if kb_path.exists():
        categories = [d.name for d in kb_path.iterdir() if d.is_dir()]
        for cat in categories:
            pdf_count = len(list((kb_path / cat).glob("*.pdf")))
            st.caption(f"{cat}: {pdf_count} PDFs")
    else:
        st.caption("No knowledge_base/ directory found.")
    if DGCA_RULES_DIR.exists():
        pdf_count = len(list(DGCA_RULES_DIR.glob("*.pdf")))
        st.caption(f"dgca_rules/: {pdf_count} PDFs")

    st.divider()
    if st.button("Show Welcome Guide"):
        st.session_state.show_welcome = True
        st.rerun()

# --- Tabs ---
tab_chat, tab_flights, tab_crew, tab_ops, tab_ml, tab_ground, tab_pax, tab_staffing, tab_revenue = st.tabs([
    "Chat (Ask Anything)", "Flights", "Crew", "Disruptions & Reports", "ML & Forecasting",
    "Ground Operations", "Passenger & Baggage", "Staffing Forecast", "Revenue & Analytics"
])

# --- Chat Tab ---
with tab_chat:
    st.info("Type a question in plain English. Try: 'Who can fly AI-501?' or 'What are the night duty rules?'")
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
                decision = route_request(prompt)
                intent = decision.get("intent", "Data_Query")

                replace_match = re.search(r"find\s+(?:new\s+)?(?:replacement|replacements|crew)\s+(?:for|of)\s+([A-Z]{2,3}[-_]?\d{2,4})", prompt, re.IGNORECASE)
                if replace_match:
                    target_fid = replace_match.group(1).upper().replace("_", "-")
                    result = find_replacement_crew(target_fid, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                    if result["status"] == "success" and result.get("candidates"):
                        msg = f"### Replacement Candidates for {target_fid}\n\n"
                        for role_name, role_data in result["candidates"].items():
                            msg += f"**{role_name}** (need {role_data['needed']}):\n"
                            for c in role_data["candidates"]:
                                if c["eligible"]:
                                    msg += f"- ✅ **{c['name']}** ({c['crew_id']}) - Duty: {c['current_duty']}h, Rolling: {c['rolling_7_day']}h - Cost: ${c['cost']:.2f}\n"
                                else:
                                    msg += f"- ❌ {c['name']} ({c['crew_id']}) - {', '.join(c['violations'])}\n"
                            if not role_data["candidates"]:
                                msg += "- No eligible candidates found in standby pool.\n"
                            msg += "\n"
                        st.markdown("### Replacement Crew")
                        st.write(msg)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "flights",
                            "content": msg, "decision": decision,
                        })
                        st.stop()
                    elif result["status"] == "success":
                        st.info(result.get("message", "No replacements needed."))
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
                        result_text = answer_flight_query(prompt)
                    st.markdown("### Flight Information")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "flights",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Data_Query":
                    result_text = answer_crew_query(prompt, str(DEFAULT_CSV_PATH))
                    st.markdown("### Crew Roster Analysis")
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
                            result = process_delay(fid, delay_minutes, str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
                            if result["status"] == "success":
                                msg = f"### Flight {fid} Delayed (+{int(delay_minutes)} min)\n\n"
                                msg += "| Crew | Role | Duty Now | After Delay | Limit | Status |\n"
                                msg += "|------|------|----------|-------------|-------|--------|\n"
                                for c in result["crew_impact"]:
                                    status_icon = "✅" if c["status"] == "ok" else "❌ Unassigned"
                                    msg += f"| {c['name']} | {c['role']} | {c['current_duty']}h | {c['projected_duty']}h | {c['duty_limit']:.0f}h | {status_icon} |\n"

                                if result["unassigned_count"] > 0:
                                    msg += f"\n⚠️ **{result['unassigned_count']} crew unassigned** (DGCA overtime violation):\n"
                                    for u in result["unassigned_crew"]:
                                        msg += f"- **{u['name']}** ({u['role']}): {u['reason']}\n"
                                    msg += f'\nAsk "find replacements for {fid}" to get new candidates.'
                                else:
                                    msg += "\n✅ All assigned crew remain compliant."

                                st.markdown("### Delay Impact Report")
                                st.write(msg)
                                st.session_state.messages.append({
                                    "role": "assistant", "type": "flights",
                                    "content": msg, "decision": decision,
                                })
                            else:
                                st.error(result.get("message", "Delay processing failed."))
                        else:
                            st.info(f"Tell me how much to delay {fid} by (e.g., 'Delay {fid} by 3 hours').")
                    else:
                        st.info("Please specify a flight ID to delay (e.g., 'Delay AI-501 by 2 hours').")

                elif intent == "Staff_Analytics":
                    from agents.ground_ops_agent import answer_ground_ops_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_ground_ops_query(prompt, flight_ids)
                    st.markdown("### Staff Analytics")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "staffing",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Passenger_Flow":
                    from agents.passenger_agent import answer_passenger_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_passenger_query(prompt, flight_ids)
                    st.markdown("### Passenger & Baggage Analysis")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "passenger",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Turnaround_Status":
                    from agents.ground_ops_agent import answer_ground_ops_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_ground_ops_query(prompt, flight_ids)
                    st.markdown("### Ground Operations")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "ground_ops",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Security_Analytics":
                    from agents.passenger_agent import answer_passenger_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_passenger_query(prompt, flight_ids)
                    st.markdown("### Security Analytics")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "security",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Revenue_Analytics":
                    from agents.passenger_agent import answer_passenger_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_passenger_query(prompt, flight_ids)
                    st.markdown("### Revenue & Analytics")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "revenue",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Maintenance_Status":
                    from agents.ground_ops_agent import answer_ground_ops_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_ground_ops_query(prompt, flight_ids)
                    st.markdown("### Maintenance Status")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "maintenance",
                        "content": result_text, "decision": decision,
                    })

                elif intent == "Recovery_Plan":
                    from agents.recovery_agent import answer_recovery_query
                    extraction = decision.get("extraction", {})
                    flight_ids = extraction.get("flight_ids", [])
                    result_text = answer_recovery_query(prompt, flight_ids)
                    st.markdown("### Recovery Plan")
                    st.write(result_text)
                    with st.expander("System Extraction & Audit Trail"):
                        st.json(decision)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "recovery",
                        "content": result_text, "decision": decision,
                    })

                else:
                    result_text = answer_crew_query(prompt, str(DEFAULT_CSV_PATH))
                    st.markdown("### Crew Roster Analysis")
                    st.write(result_text)
                    st.session_state.messages.append({
                        "role": "assistant", "type": "data",
                        "content": result_text, "decision": decision,
                    })

            except Exception as exc:
                st.error(f"Dispatcher failed: {exc}")

# --- Flights Tab ---
with tab_flights:
    st.header("Flight Schedule")
    st.caption("View all flights, add new ones, or filter by airport.")

    with st.expander("Add New Flight", expanded=False):
        st.caption("Fill in the details below to add a new flight to the schedule.")
        with st.form("add_flight"):
            c1, c2, c3 = st.columns(3)
            fid = c1.text_input("Flight ID", "AI-401")
            origin = c2.text_input("Origin", "DEL")
            dest = c3.text_input("Destination", "BOM")
            c4, c5, c6 = st.columns(3)
            dep_time = c4.time_input("Departure Time", datetime.now().time())
            ac_type = c5.selectbox("Aircraft Type", ["B737", "A320", "A321", "ATR"])
            pax = c6.number_input("Passengers", 0, 300, 160)
            c7, c8, c9 = st.columns(3)
            gate = c7.text_input("Gate", "A01")
            terminal = c8.text_input("Terminal", "T1")
            duration = c9.number_input("Duration (min)", 30, 600, 120)
            is_intl = st.checkbox("International")
            submitted = st.form_submit_button("Add Flight")
            if submitted:
                today = datetime.now().replace(hour=dep_time.hour, minute=dep_time.minute, second=0, microsecond=0)
                new_flight = Flight(
                    flight_id=fid.upper(), origin=origin.upper(), destination=dest.upper(),
                    std=today, aircraft_type=ac_type.upper(),
                    gate=gate, terminal=terminal, pax_count=pax,
                    flight_duration_min=duration, is_international=is_intl,
                )
                insert_flight(new_flight, DEFAULT_DB_PATH)
                st.success(f"Flight {fid.upper()} added.")

                assignment_result = auto_assign_flight(
                    fid.upper(), str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH
                )
                if assignment_result["assigned_count"] > 0:
                    st.info(f"Auto-assigned {assignment_result['assigned_count']} crew members to {fid.upper()}.")
                    for a in assignment_result["assignments"]:
                        st.write(f"  - {a['name']} ({a['role']})")
                elif assignment_result["already_assigned"] > 0:
                    st.info(f"{fid.upper()} already has {assignment_result['already_assigned']} crew assigned.")
                else:
                    st.warning(f"No eligible crew found for {fid.upper()}. Check standby roster.")
                st.rerun()

    origin_tab = st.selectbox("Filter by Origin", ["All", "DEL", "BOM", "CCU", "BLR", "MAA", "HYD"], key="flights_origin")
    flights = get_flights(
        db_path=DEFAULT_DB_PATH,
        origin=origin_tab if origin_tab != "All" else None,
    )
    if flights:
        flight_data = [f.to_dict() for f in flights]
        df = pd.DataFrame(flight_data)
        display_cols = [c for c in df.columns if c not in ["disruption_reason"]]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No flights in schedule. Use 'Load Sample Flights' in the sidebar to populate.")

# --- Crew Tab ---
with tab_crew:
    st.header("Crew Management")
    st.caption("View crew roster and check who is eligible for specific flight scenarios.")
    crew = load_crew(DEFAULT_CSV_PATH)
    if crew:
        crew_data = [m.to_dict() for m in crew]
        df = pd.DataFrame(crew_data)
        qual_cols = [c for c in df.columns if c not in ["qualifications"]]
        display_df = df[qual_cols].copy()
        display_df["qualifications"] = df["qualifications"].apply(
            lambda q: ", ".join(x.get("aircraft_type", "") for x in q) if q else "None"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Eligibility Checker")
        st.caption("Test which crew members are legal for a specific flight scenario.")
        check_col1, check_col2 = st.columns(2)
        with check_col1:
            flight_hours = st.number_input("Scenario Flight Hours", 0.5, 16.0, 3.0, 0.5)
        with check_col2:
            is_night = st.checkbox("Night Duty")
        if st.button("Run Batch Eligibility Check"):
            result = batch_validate(
                str(DEFAULT_CSV_PATH),
                scenario_flight_hours=flight_hours,
                scenario_is_night_duty=is_night,
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Total", result["total"])
            c2.metric("Eligible", result["eligible"])
            c3.metric("Ineligible", result["ineligible"])
            details = result["details"]
            ineligible = {k: v for k, v in details.items() if not v["eligible"]}
            if ineligible:
                st.subheader("Ineligible Crew")
                inel_df = pd.DataFrame([
                    {"ID": k, "Name": v["name"], "Role": v["role"],
                     "Violations": "; ".join(v["violations"])}
                    for k, v in ineligible.items()
                ])
                st.dataframe(inel_df, hide_index=True)

        st.divider()
        st.subheader("Create New Staff")
        st.caption("Add new crew members to the roster.")
        with st.form("create_staff"):
            c1, c2, c3 = st.columns(3)
            new_id = c1.text_input("Crew ID", "CRW023")
            new_name = c2.text_input("Full Name", "New Crew Member")
            new_role = c3.selectbox("Role", ["Captain", "FO", "CabinCrew", "GroundStaff"])
            c4, c5, c6 = st.columns(3)
            new_base = c4.text_input("Base Airport", "DEL")
            new_quals = c5.text_input("Qualifications (semicolon-separated)", "B737;A320")
            new_cost = c6.number_input("Base Cost", 40.0, 500.0, 100.0, 10.0)
            create_btn = st.form_submit_button("Create Staff")
            if create_btn:
                result = create_staff(
                    str(DEFAULT_CSV_PATH), new_id, new_name, new_role,
                    new_base, new_quals, new_cost,
                )
                if result["status"] == "success":
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result["message"])

        st.divider()
        st.subheader("Standby vs Assigned")
        st.caption("See which crew are available (standby) vs assigned to flights.")
        summary = get_assignment_summary(str(DEFAULT_CSV_PATH), DEFAULT_DB_PATH)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Crew", summary["total_crew"])
        c2.metric("Assigned", summary["assigned_count"])
        c3.metric("Standby", summary["standby_count"])

        if summary["assigned_crew"]:
            st.subheader("Assigned Crew")
            assigned_df = pd.DataFrame(summary["assigned_crew"])
            st.dataframe(assigned_df, use_container_width=True, hide_index=True)

        if summary["standby_crew"]:
            st.subheader("Standby Crew")
            standby_df = pd.DataFrame(summary["standby_crew"])
            st.dataframe(standby_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No crew data found.")

# --- Operations Tab ---
with tab_ops:
    st.header("Disruptions & Reports")
    st.caption("Handle delayed flights, find replacement crew, and generate compliance reports.")

    st.subheader("Disruption Management")
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
        st.info("No disrupted flights. Load sample flights and mark some as delayed to test.")

    st.subheader("Upcoming Flights")
    st.caption("Flights departing in the next 12 hours.")
    upcoming = get_upcoming_flights(hours_ahead=12, db_path=DEFAULT_DB_PATH)
    if upcoming:
        for f in upcoming:
            st.write(f"**{f.flight_id}** {f.origin}->{f.destination} at {f.std.strftime('%H:%M')} ({f.aircraft_type}) - {f.status.value}")
    else:
        st.info("No upcoming flights in the next 12 hours.")

    st.subheader("Batch Compliance Report")
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

    st.subheader("Flight Statistics")
    st.caption("Overview of flights by status and aircraft type.")
    stats = get_flight_stats(DEFAULT_DB_PATH)
    if stats["total"] > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.json(stats["by_status"])
        with col2:
            st.json(stats["by_aircraft"])

# --- ML & Forecasting Tab ---
with tab_ml:
    st.header("ML & Forecasting")
    st.caption("Predict delays, forecast demand, and optimize crew utilization using data-driven insights.")

    ml_subtab1, ml_subtab2, ml_subtab3 = st.tabs([
        "Delay Prediction", "Demand Forecasting", "Resource Optimization"
    ])

    # --- Delay Prediction ---
    with ml_subtab1:
        st.subheader("Flight Delay Predictor")
        st.caption("Estimate delay risk for a flight based on route, time, and aircraft type.")

        with st.form("delay_predict"):
            c1, c2, c3 = st.columns(3)
            pred_origin = c1.selectbox("Origin", ["DEL", "BOM", "CCU", "BLR", "MAA", "HYD"], key="pred_orig")
            pred_dest = c2.selectbox("Destination", ["BOM", "DEL", "CCU", "BLR", "MAA", "HYD"], key="pred_dest")
            pred_ac = c3.selectbox("Aircraft", ["B737", "A320", "A321", "ATR"], key="pred_ac")
            c4, c5, c6 = st.columns(3)
            pred_hour = c4.slider("Departure Hour", 0, 23, 10)
            pred_pax = c5.number_input("Passengers", 50, 300, 160)
            pred_duration = c6.number_input("Duration (min)", 30, 600, 120)
            pred_intl = st.checkbox("International Flight", key="pred_intl")
            predict_btn = st.form_submit_button("Predict Delay Risk")

        if predict_btn:
            result = predict_delay(
                origin=pred_origin, destination=pred_dest,
                aircraft_type=pred_ac, departure_hour=pred_hour,
                pax_count=pred_pax, flight_duration_min=pred_duration,
                is_international=pred_intl,
            )
            c1, c2, c3 = st.columns(3)
            prob_pct = result["delay_probability"] * 100
            c1.metric("Delay Probability", f"{prob_pct:.1f}%")
            c2.metric("Expected Delay", f"{result['expected_delay_min']} min")
            c3.metric("Risk Level", result["risk_level"])

            if result["factors"]:
                st.subheader("Contributing Factors")
                for factor in result["factors"]:
                    st.write(f"- {factor}")

            with st.expander("Model Features"):
                st.json(result["features"])

        st.divider()
        st.subheader("Historical Delay Insights")
        if st.button("Analyze Flight Data for Delays"):
            insights = get_delay_insights(DEFAULT_DB_PATH)
            if "message" in insights:
                st.info(insights["message"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flights", insights["total_flights"])
                c2.metric("Delayed", insights["delayed_flights"])
                c3.metric("Delay Rate", f"{insights['delay_rate']:.1%}")

                if insights["airport_risk_scores"]:
                    st.subheader("Airport Risk Scores")
                    risk_df = pd.DataFrame([
                        {"Airport": k, "Risk Score": v}
                        for k, v in sorted(insights["airport_risk_scores"].items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(risk_df, hide_index=True)

    # --- Demand Forecasting ---
    with ml_subtab2:
        st.subheader("Passenger Demand Forecast")
        st.caption("Predict passenger demand for any route and plan flights accordingly.")

        with st.form("demand_forecast"):
            c1, c2 = st.columns(2)
            dem_origin = c1.selectbox("Origin", ["DEL", "BOM", "CCU", "BLR", "MAA", "HYD"], key="dem_orig")
            dem_dest = c2.selectbox("Destination", ["BOM", "DEL", "CCU", "BLR", "MAA", "HYD"], key="dem_dest")
            forecast_btn = st.form_submit_button("Forecast Demand")

        if forecast_btn:
            forecast = forecast_demand(dem_origin, dem_dest)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Daily Passengers", forecast["total_daily_pax"])
            c2.metric("Flights Needed", forecast["flights_needed"])
            c3.metric("Avg Load Factor", f"{forecast['avg_load_factor']:.0%}")

            st.subheader("Peak Hours")
            peak_df = pd.DataFrame(forecast["peak_hours"])
            st.dataframe(peak_df, hide_index=True)

            st.subheader("Hourly Demand Breakdown")
            hourly_df = pd.DataFrame(forecast["hourly_breakdown"])
            st.bar_chart(hourly_df.set_index("hour")["estimated_pax"])

        st.divider()
        st.subheader("Current Demand Summary")
        if st.button("Analyze Current Demand"):
            summary = get_demand_summary(DEFAULT_DB_PATH)
            if "message" in summary:
                st.info(summary["message"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flights", summary["total_flights"])
                c2.metric("Total Passengers", summary["total_pax"])
                c3.metric("Avg Pax/Flight", summary["avg_pax_per_flight"])

                if summary["route_breakdown"]:
                    st.subheader("Route Breakdown")
                    route_df = pd.DataFrame([
                        {"Route": k, "Flights": v["flights"], "Total Pax": v["total_pax"], "Avg Pax": v["avg_pax"]}
                        for k, v in summary["route_breakdown"].items()
                    ])
                    st.dataframe(route_df, hide_index=True)

    # --- Resource Optimization ---
    with ml_subtab3:
        st.subheader("Crew Resource Optimization")
        st.caption("Score crew utilization, find optimal swaps, and forecast staffing needs.")

        if st.button("Generate Full Augmentation Report"):
            report = get_augmentation_report(DEFAULT_CSV_PATH, DEFAULT_DB_PATH)

            st.subheader("Crew Utilization Scores")
            scores_df = pd.DataFrame(report["crew_scores"])
            if not scores_df.empty:
                st.dataframe(scores_df, use_container_width=True, hide_index=True)

            st.subheader("Alerts")
            alerts = report["alerts"]
            if alerts["high_fatigue_count"] > 0:
                st.warning(f"{alerts['high_fatigue_count']} crew members with HIGH fatigue risk")
                st.write(", ".join(alerts["high_fatigue_crew"]))
            if alerts["underutilized_count"] > 0:
                st.info(f"{alerts['underutilized_count']} crew members are underutilized")
            if alerts["idle_count"] > 0:
                st.info(f"{alerts['idle_count']} crew members are currently idle")

            if report["top_swaps"]:
                st.subheader("Recommended Crew Swaps")
                swaps_df = pd.DataFrame(report["top_swaps"])
                st.dataframe(swaps_df, hide_index=True)

            st.subheader("Staffing Forecast")
            forecast = report["forecast"]
            if "error" not in forecast:
                c1, c2, c3 = st.columns(3)
                c1.metric("Flights Today", forecast["flights_today"])
                c2.metric("Total Flight Hours", forecast["total_flight_hours"])
                c3.metric("Expected Disruptions", forecast["expected_disruptions"])

                st.subheader("Role Breakdown")
                for role, data in forecast["role_breakdown"].items():
                    status_color = "normal" if data["status"] == "Sufficient" else "inverse"
                    st.metric(
                        f"{role} ({data['status']})",
                        f"Available: {data['available']}",
                        f"Required: {data['required']}",
                        delta=f"Gap: {data['gap']}" if data["gap"] > 0 else None,
                    )

        st.divider()
        st.subheader("Quick Crew Scoring")
        if st.button("Score All Crew"):
            scores = score_crew_utilization(DEFAULT_CSV_PATH)
            scores_df = pd.DataFrame(scores)
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

# --- Ground Operations Tab ---
with tab_ground:
    st.header("Ground Operations")
    st.caption("Boarding, turnaround, baggage pipeline, and maintenance analytics from the operations dataset.")

    ops = get_ops_summary(DEFAULT_DB_PATH)
    if not ops.get("loaded"):
        st.info("Load the Operations Dataset from the sidebar first.")
    else:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Flights", ops.get("ops_flights", 0))
        g2.metric("Gate Events", ops.get("ops_gate_events", 0))
        g3.metric("Baggage", ops.get("ops_baggage", 0))
        g4.metric("Maintenance", ops.get("ops_maintenance", 0))

        st.subheader("Boarding Efficiency")
        be = get_boarding_efficiency(DEFAULT_DB_PATH)
        if be.get("loaded") and be.get("flights_analyzed", 0) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("Flights Analyzed", be["flights_analyzed"])
            c2.metric("Avg Boarding Time", f"{be['avg_boarding_min']:.0f} min")
            c3.metric("Efficiency/Pax", f"{be['efficiency_per_pax']:.1f} min/pax")
        else:
            st.info("No boarding data available.")

        st.subheader("Turnaround Profile")
        turnaround_fid = st.text_input("Flight ID for turnaround", "6E-1311", key="turnaround_fid")
        if st.button("Get Turnaround Profile"):
            tp = get_turnaround_profile(DEFAULT_DB_PATH, turnaround_fid)
            if tp.get("loaded") and not tp.get("error"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Gate Events", len(tp.get("gate_events", [])))
                c2.metric("Bags", tp.get("baggage_summary", {}).get("total_bags", 0))
                c3.metric("Boarding Duration", f"{tp.get('boarding_duration_min', 0)} min")
                c4.metric("Airworthy", "Yes" if tp.get("airworthy") else "No")
                if tp.get("baggage_summary"):
                    st.write(f"Baggage: {tp['baggage_summary']['total_weight_kg']}kg total, {tp['baggage_summary']['total_pieces']} pieces")
            elif tp.get("error"):
                st.error(tp["error"])

        st.subheader("Maintenance Impact")
        mi = get_maintenance_impact(DEFAULT_DB_PATH)
        if mi.get("loaded"):
            c1, c2 = st.columns(2)
            c1.metric("Total Work Orders", mi.get("total_work_orders", 0))
            top_defects = mi.get("top_defects", [])
            if top_defects:
                defect_df = pd.DataFrame(top_defects)
                st.dataframe(defect_df, use_container_width=True, hide_index=True)

# --- Passenger & Baggage Tab ---
with tab_pax:
    st.header("Passenger & Baggage")
    st.caption("Passenger demand, baggage load, nationality demographics, and connecting pax flow.")

    if not ops.get("loaded") if 'ops' in dir() else True:
        ops = get_ops_summary(DEFAULT_DB_PATH)
    if not ops.get("loaded"):
        st.info("Load the Operations Dataset from the sidebar first.")
    else:
        st.subheader("Passenger Profile")
        pax_fid = st.text_input("Flight ID (or leave empty for all)", "", key="pax_fid")
        if st.button("Get Passenger Profile"):
            fid = pax_fid if pax_fid.strip() else None
            pp = get_passenger_profile(DEFAULT_DB_PATH, fid)
            if pp.get("loaded"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Passengers", pp["total_passengers"])
                c2.metric("Connecting", pp["connecting_passengers"])
                c3.metric("Connecting Rate", f"{pp['connecting_rate']:.1%}")
                c4.metric("Avg Loyalty", f"{pp['avg_loyalty_score']:.1f}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write("**Nationality**")
                    for nat, cnt in sorted(pp["nationality_distribution"].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"  {nat}: {cnt}")
                with col_b:
                    st.write("**Class**")
                    for cls, cnt in pp["class_distribution"].items():
                        st.write(f"  {cls}: {cnt}")
                with col_c:
                    st.write("**Age Category**")
                    for age, cnt in pp["age_distribution"].items():
                        if age:
                            st.write(f"  {age}: {cnt}")

        st.subheader("Baggage Load Profile")
        bag_fid = st.text_input("Flight ID for baggage", "6E-1311", key="bag_fid")
        if st.button("Get Baggage Profile"):
            bl = get_baggage_load_profile(DEFAULT_DB_PATH, bag_fid)
            if bl.get("loaded"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Bags", bl["total_bags"])
                c2.metric("Total Weight", f"{bl['total_weight_kg']:.0f} kg")
                c3.metric("Avg Weight", f"{bl['avg_weight_per_bag']:.1f} kg")
                c4.metric("Delayed", bl["delayed_bags"])

        st.subheader("Demand by Route")
        dr = get_demand_by_route(DEFAULT_DB_PATH)
        if dr.get("loaded") and dr.get("routes"):
            route_df = pd.DataFrame([
                {"Route": k, "Flights": v["flights"], "Total Pax": v["total_pax"],
                 "Avg Pax/Flight": v["avg_pax_per_flight"], "Route Type": v["route_type"]}
                for k, v in sorted(dr["routes"].items(), key=lambda x: x[1]["total_pax"], reverse=True)
            ])
            st.dataframe(route_df, use_container_width=True, hide_index=True)
        else:
            st.info("No route demand data available.")

# --- Staffing Forecast Tab ---
with tab_staffing:
    st.header("Staffing Forecast")
    st.caption("Staff role distribution, shift coverage analysis, and understaffed period alerts.")

    if not ops.get("loaded") if 'ops' in dir() else True:
        ops = get_ops_summary(DEFAULT_DB_PATH)
    if not ops.get("loaded"):
        st.info("Load the Operations Dataset from the sidebar first.")
    else:
        st.subheader("Staff Role Distribution")
        dist = get_staff_role_distribution(DEFAULT_DB_PATH)
        if dist.get("loaded"):
            c1, c2 = st.columns(2)
            c1.metric("Total Staff", dist["total_staff"])
            role_df = pd.DataFrame([
                {"Role": k, "Count": v, "Fraction": f"{dist['role_fractions'].get(k, 0):.1%}"}
                for k, v in dist["by_role"].items()
            ])
            c2.dataframe(role_df, use_container_width=True, hide_index=True)

        st.subheader("Shift Coverage Analysis")
        if st.button("Analyze Shift Coverage"):
            cov = get_shift_coverage_analysis(DEFAULT_DB_PATH)
            if cov.get("loaded") and cov.get("coverage_by_hour"):
                cov_df = pd.DataFrame(cov["coverage_by_hour"])
                display_cols = [c for c in cov_df.columns if c != "staff_by_role"]
                st.dataframe(cov_df[display_cols], use_container_width=True, hide_index=True)

                understaffed = identify_understaffed_periods(DEFAULT_DB_PATH)
                if understaffed:
                    st.warning(f"{len(understaffed)} understaffed period(s) detected")
                    under_df = pd.DataFrame(understaffed)
                    st.dataframe(under_df, use_container_width=True, hide_index=True)
                else:
                    st.success("No understaffed periods detected")

        st.subheader("Staff Utilization")
        if st.button("Check Staff Utilization"):
            util = get_staff_utilization(DEFAULT_DB_PATH)
            if util.get("loaded"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Staff", util["total_staff"])
                c2.metric("Assigned to Gate", util["assigned_to_gate_events"])
                c3.metric("Assigned to Security", util["assigned_to_security"])
                c4.metric("Idle", util["idle_count"])

# --- Revenue & Analytics Tab ---
with tab_revenue:
    st.header("Revenue & Analytics")
    st.caption("Retail revenue, delay cause breakdown, security throughput, and queue predictions.")

    if not ops.get("loaded") if 'ops' in dir() else True:
        ops = get_ops_summary(DEFAULT_DB_PATH)
    if not ops.get("loaded"):
        st.info("Load the Operations Dataset from the sidebar first.")
    else:
        r_subtab1, r_subtab2, r_subtab3 = st.tabs(["Retail Revenue", "Delay Analytics", "Security"])

        with r_subtab1:
            st.subheader("Revenue by Flight")
            rf = get_revenue_by_flight(DEFAULT_DB_PATH)
            if rf.get("loaded"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Revenue", f"INR {rf['total_revenue']:,.0f}")
                c2.metric("Transactions", rf["total_transactions"])
                c3.metric("Avg per Txn", f"INR {rf['avg_revenue_per_txn']:,.0f}")

                if rf.get("top_flights"):
                    rev_df = pd.DataFrame(rf["top_flights"])
                    st.dataframe(rev_df, use_container_width=True, hide_index=True)

            st.subheader("Spend by Passenger Class")
            ps = get_passenger_spend_profile(DEFAULT_DB_PATH)
            if ps.get("loaded") and ps.get("by_class"):
                spend_df = pd.DataFrame(ps["by_class"])
                st.dataframe(spend_df, use_container_width=True, hide_index=True)

        with r_subtab2:
            st.subheader("Delay Cause Breakdown")
            cb = get_delay_cause_breakdown(DEFAULT_DB_PATH)
            if cb.get("causes"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Flights", cb["total_flights"])
                c2.metric("Delayed", cb["delayed_flights"])
                c3.metric("Delay Rate", f"{cb['overall_delay_rate']:.1%}")

                cause_df = pd.DataFrame([
                    {"Cause": k, "Count": v["count"], "Avg Delay (min)": v["avg_delay_min"]}
                    for k, v in cb["causes"].items()
                ])
                st.dataframe(cause_df, use_container_width=True, hide_index=True)

            st.subheader("Delay by Airport")
            da = get_delay_by_airport(DEFAULT_DB_PATH)
            if isinstance(da, dict) and da and not da.get("message"):
                ap_df = pd.DataFrame([
                    {"Airport": k, "Total": v["total"], "Delayed": v["delayed"],
                     "Delay Rate": f"{v['delay_rate']:.1%}", "Avg Delay (min)": v["avg_delay_min"]}
                    for k, v in da.items()
                ])
                st.dataframe(ap_df, use_container_width=True, hide_index=True)

            st.subheader("Delay by Route Type")
            dr = get_delay_by_route_type(DEFAULT_DB_PATH)
            if isinstance(dr, dict) and dr.get("by_route_type"):
                route_delay_df = pd.DataFrame([
                    {"Route Type": k, "Total": v["total"], "Delayed": v["delayed"],
                     "Delay Rate": f"{v['delay_rate']:.1%}"}
                    for k, v in dr["by_route_type"].items()
                ])
                st.dataframe(route_delay_df, use_container_width=True, hide_index=True)

        with r_subtab3:
            st.subheader("Security Throughput")
            sth = get_security_throughput(DEFAULT_DB_PATH)
            if sth.get("loaded"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Screenings", sth["total_screenings"])
                c2.metric("Avg Processing Time", f"{sth['avg_processing_time']:.0f}s")
                c3.metric("Avg Pass/Hour", f"{sth['avg_pass_per_hour']:.0f}")

                if sth.get("by_screen_type"):
                    st.write("**By Screen Type**")
                    type_df = pd.DataFrame(sth["by_screen_type"])
                    st.dataframe(type_df, use_container_width=True, hide_index=True)

            st.subheader("Queue Prediction")
            if st.button("Predict Queue Buildup"):
                pq = predict_queue_buildup(DEFAULT_DB_PATH)
                if pq.get("loaded") and pq.get("predictions"):
                    pq_df = pd.DataFrame(pq["predictions"])
                    high_risk = pq_df[pq_df["risk_level"] == "High"]
                    if not high_risk.empty:
                        st.warning(f"{len(high_risk)} high-risk hours detected")
                    st.bar_chart(pq_df.set_index("hour")["predicted_wait_min"])
