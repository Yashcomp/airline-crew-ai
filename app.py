from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from rag_engine import retrieve_legal_guidance
from solver import DEFAULT_SCENARIO_FLIGHT_HOURS, solve_from_csv


class DummyLLM:
    def synthesize(self, scenario_text: str, legal_text: str, solver_result: dict) -> str:
        return (
            "Mock LLM response: the router, retrieval path, and optimizer all executed successfully. "
            "This is a placeholder until Azure OpenAI credentials are configured."
        )


def get_llm() -> DummyLLM:
    return DummyLLM()


def default_csv_path() -> Path:
    return Path(__file__).with_name("crew_standby_list.csv")


st.set_page_config(page_title="Airline Crew Scheduling Prototype", layout="wide")
st.title("Airline Crew Scheduling Prototype")
st.caption("Mock router + legal retrieval stub + PuLP assignment solver")

scenario_text = st.text_area(
    "Scenario",
    value="Flight delayed by 2 hours. Need a legal standby crew assignment.",
    height=120,
)

flight_hours = st.number_input(
    "Scenario flight hours",
    min_value=0.5,
    max_value=20.0,
    value=float(DEFAULT_SCENARIO_FLIGHT_HOURS),
    step=0.5,
)

csv_file = st.file_uploader("Optional crew CSV override", type=["csv"])

if st.button("Run Prototype"):
    if csv_file is not None:
        csv_path = default_csv_path().with_name("uploaded_crew_standby_list.csv")
        csv_path.write_bytes(csv_file.getbuffer())
    else:
        csv_path = default_csv_path()

    legal_text = retrieve_legal_guidance(scenario_text)
    solver_result = solve_from_csv(csv_path, scenario_flight_hours=float(flight_hours))
    llm = get_llm()
    final_response = llm.synthesize(scenario_text, legal_text, solver_result)

    st.subheader("Legal Guidance")
    st.write(legal_text)

    st.subheader("Solver Output")
    st.json(solver_result)

    st.subheader("Synthesized Response")
    st.write(final_response)

    st.subheader("Structured Debug Payload")
    st.code(json.dumps({"scenario": scenario_text, "legal": legal_text, "solver": solver_result}, indent=2), language="json")