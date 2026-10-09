"""
test_eligibility.py — Unit tests for src.eligibility .
"""

from __future__ import annotations

import pytest

from src.models import Candidate, ParseStatus
from src.eligibility import check_eligibility


# ===========================================================================
# Unit tests: check_eligibility
# ===========================================================================

def test_eligible_python_and_strong_ai():
    c = Candidate(
        source_file="test.pdf",
        skills=["Python", "LangChain"],
        projects=["Built an agentic workflow using LangChain and Python."],
        extracted_text="Python and LangChain developer."
    )
    res = check_eligibility(c)
    assert res.eligible is True
    assert "Python" in res.matched_skills
    assert "LangChain" in res.matched_skills
    assert len(res.rejection_reasons) == 0
    assert len(res.python_evidence) > 0
    assert len(res.ai_evidence) > 0


def test_rejected_python_without_ai():
    c = Candidate(
        source_file="test.pdf",
        skills=["Python", "Django", "JavaScript"],
        extracted_text="I developed web apps with Python, Django and JavaScript."
    )
    res = check_eligibility(c)
    assert res.eligible is False
    assert len(res.rejection_reasons) == 1
    assert "No meaningful AI" in res.rejection_reasons[0]
    assert len(res.python_evidence) > 0
    assert len(res.ai_evidence) == 0


def test_rejected_ai_without_python():
    c = Candidate(
        source_file="test.pdf",
        skills=["JavaScript", "LangChain"],
        extracted_text="I developed AI apps with LangChain and Node.js."
    )
    res = check_eligibility(c)
    assert res.eligible is False
    assert len(res.rejection_reasons) == 1
    assert "No Python evidence" in res.rejection_reasons[0]
    assert len(res.python_evidence) == 0
    assert len(res.ai_evidence) > 0


def test_eligible_with_javascript_alongside():
    c = Candidate(
        source_file="test.pdf",
        skills=["Python", "LangGraph", "JavaScript", "React"],
        extracted_text="Full stack engineer using React, JavaScript, Python, and LangGraph for agentic systems."
    )
    res = check_eligibility(c)
    assert res.eligible is True
    assert "Python" in res.matched_skills
    assert "LangGraph" in res.matched_skills


def test_case_insensitive_matching():
    c = Candidate(
        source_file="test.pdf",
        extracted_text="i am good at pYtHoN and rAg pipeline construction."
    )
    res = check_eligibility(c)
    assert res.eligible is True
    assert "Python" in res.matched_skills
    assert "RAG pipeline" in res.matched_skills


def test_framework_names_without_meaningful_project_evidence_rejected():
    # Only GenAI is mentioned, which is a contextual keyword but lacks a strong context verb or co-occurrence
    c = Candidate(
        source_file="test.pdf",
        skills=["Python"],
        extracted_text="Python developer interested in GenAI."
    )
    res = check_eligibility(c)
    assert res.eligible is False
    assert len(res.ai_evidence) == 0


def test_contextual_ai_with_meaningful_project_verb_eligible():
    c = Candidate(
        source_file="test.pdf",
        skills=["Python"],
        extracted_text="Developed a customer support chatbot using GenAI and Python."
    )
    res = check_eligibility(c)
    assert res.eligible is True
    assert len(res.ai_evidence) > 0


def test_multiple_contextual_ai_eligible():
    # Co-occurrence of multiple contextual keywords
    c = Candidate(
        source_file="test.pdf",
        skills=["Python"],
        extracted_text="Python developer working with LLM and RAG."
    )
    res = check_eligibility(c)
    assert res.eligible is True
    assert len(res.ai_evidence) > 0


def test_parse_failure_rejected():
    c = Candidate(
        source_file="test.pdf",
        parse_status=ParseStatus.FAILURE,
        parse_error="Failed to extract text."
    )
    res = check_eligibility(c)
    assert res.eligible is False
    assert len(res.rejection_reasons) == 1
    assert "could not be parsed" in res.rejection_reasons[0]


def test_missing_or_ambiguous_fields():
    c = Candidate(
        source_file="test.pdf",
        skills=[], # empty
        projects=[], # empty
        extracted_text="Python and LlamaIndex" # present in text
    )
    res = check_eligibility(c)
    assert res.eligible is True

