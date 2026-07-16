# Airline Crew Scheduling Prototype

## Overview

This repository contains a local, source-only prototype for an airline crew scheduling system built around a strict Agentic Router pattern:

- `app.py` is the Streamlit frontend used for local interaction.
- `rag_engine.py` handles the rules path and retrieves DGCA FDTL guidance from the PDFs stored in `dgca_rules/`.
- `solver.py` handles the deterministic optimization path using PuLP.
- The router/orchestrator pattern keeps document retrieval and mathematical optimization separate, then combines their outputs in the synthesizer layer.

The design is intentionally split into two independent paths:

1. **Rules Path (RAG)** - Loads DGCA PDFs, chunks them locally, indexes them in FAISS, and retrieves relevant clauses.
2. **Math Path (PuLP)** - Applies hard operational constraints to select the cheapest legal crew assignment.

The offline behavior is safe for restricted corporate systems. When Azure OpenAI credentials are missing, the app uses a `DummyLLM` fallback and synthetic embeddings so the UI and solver still run without internet access or model downloads.

## Architecture

### Agentic Router

The intended production flow is:

1. A user enters a disruption scenario in Streamlit.
2. The router decides whether to query DGCA rules, solve the assignment problem, or both.
3. The DGCA path retrieves exact clauses from the local PDF corpus.
4. The PuLP path computes a legal assignment from the crew CSV.
5. A final synthesizer combines the retrieved rule text and solver output into a human-readable response.

### RAG Path

`rag_engine.py` uses:

- `PyPDFDirectoryLoader` to load PDFs from `dgca_rules/`
- `RecursiveCharacterTextSplitter` with chunk size 1000 and overlap 150
- `FAISS` for the local vector store
- `AzureOpenAIEmbeddings` and `AzureChatOpenAI` only when Azure credentials are present
- `FakeEmbeddings` plus a `DummyLLM` fallback when running offline

### Optimization Path

`solver.py` uses PuLP to select the cheapest legal crew combination from `crew_standby_list.csv`.

## DGCA Constraints Enforced in the Solver

The current prototype enforces these operational constraints:

- Maximum duty period: 12 hours
- Maximum rolling 7-day hours: 35 hours
- Minimum rank coverage: at least 1 Captain
- Additional prototype coverage: FO, CabinCrew, and GroundStaff counts configured in the solver defaults
- Rest status must be legal
- Consecutive night duties: a crew member cannot be assigned if the new duty would exceed 2 consecutive night duties in the 168-hour window

The CSV schema expected by the solver includes:

- `Crew_ID`
- `Name`
- `Rank`
- `Role`
- `Current_Duty_Hours`
- `Rolling_7_Day_Hours`
- `Consecutive_Night_Shifts`
- `Rest_Status`
- `Cost_Multiplier`

## Corporate Offline Setup

This project is designed for a locked-down Windows environment with no Docker and no WSL.

### Recommended workflow

1. Copy the source folder to the target machine.
2. Unzip it into a writable local directory.
3. Create a fresh Python virtual environment on that machine.
4. Install dependencies from `requirements.txt`.
5. Place the DGCA PDFs inside `dgca_rules/`.
6. Run the Streamlit app locally.

### Offline fallback behavior

- If Azure OpenAI settings are not present, the app uses `DummyLLM`.
- If Azure embeddings are not available, the RAG path falls back to `FakeEmbeddings`.
- No cloud vector database is used.
- No Docker, WSL, or background model downloads are required.

### Important environment variables for Azure mode

Set these only when the corporate machine is allowed to use Azure OpenAI:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`

## Local Run

After installing dependencies:

```powershell
streamlit run app.py
```

If Streamlit is not on PATH:

```powershell
python -m streamlit run app.py
```

## Repository Layout

- `app.py` - Streamlit UI and top-level orchestration
- `rag_engine.py` - DGCA PDF retrieval pipeline
- `solver.py` - PuLP optimization engine
- `crew_standby_list.csv` - Prototype input data
- `dgca_rules/` - Local DGCA PDFs
- `requirements.txt` - Python dependencies

## GitHub Deployment

This project is intended to be published as a source-only repository named `airline-crew-ai` under `https://github.com/Yashcomp`.

## Notes

- Keep the repository source-only for transferability.
- Do not commit `.venv`, `__pycache__`, or generated vector indexes.
- Rebuild the FAISS index locally from the PDFs when needed.