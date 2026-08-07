from __future__ import annotations

from rag_engine import _categorize_query, retrieve_legal_guidance


def test_categorize_regulations():
    assert _categorize_query("What is the DGCA duty time limit?") == "regulations"


def test_categorize_sop():
    assert _categorize_query("What is the SOP for gate operations?") == "sops"


def test_categorize_unknown():
    assert _categorize_query("How many flights depart today?") is None


def test_retrieve_legal_guidance_empty_folder(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_engine._guide_definition_lookup", lambda q: None)
    answer = retrieve_legal_guidance(
        "How many flights depart today?", pdf_folder=tmp_path
    )
    assert isinstance(answer, str)
    assert answer.strip()
