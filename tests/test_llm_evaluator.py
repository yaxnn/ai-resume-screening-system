"""
Tests for Phase 4 LLM Evaluator.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from src.llm_evaluator import evaluate_candidate_llm, LLMEvaluation

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

@pytest.fixture
def mock_no_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

def test_evaluate_missing_api_key(mock_no_env):
    res = evaluate_candidate_llm("test text")
    assert res is None

@patch("src.llm_evaluator.genai.Client")
def test_evaluate_success(mock_client, mock_env):
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    
    # Mocking response
    mock_response = MagicMock()
    mock_response.text = '''{
        "ai_project_score": 35,
        "python_backend_score": 25,
        "cloud_fullstack_score": 10,
        "engineering_depth_score": 5,
        "strengths": ["Strong RAG"],
        "concerns": ["None"],
        "project_summary": "Built AI tools",
        "evidence_citations": ["Developed RAG"],
        "confidence_level": "High"
    }'''
    mock_instance.models.generate_content.return_value = mock_response

    res = evaluate_candidate_llm("test resume text")
    assert res is not None
    assert res.ai_project_score == 35
    assert res.project_summary == "Built AI tools"

@patch("src.llm_evaluator.genai.Client")
def test_evaluate_invalid_json(mock_client, mock_env):
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    
    mock_response = MagicMock()
    mock_response.text = '{"ai_project_score": 35, invalid json'
    mock_instance.models.generate_content.return_value = mock_response

    # It should retry and eventually fail
    res = evaluate_candidate_llm("test resume text", max_retries=2)
    assert res is None
    assert mock_instance.models.generate_content.call_count == 2

@patch("src.llm_evaluator.genai.Client")
def test_evaluate_api_exception(mock_client, mock_env):
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    
    mock_instance.models.generate_content.side_effect = Exception("API Rate Limit")

    res = evaluate_candidate_llm("test resume text", max_retries=2)
    assert res is None
    assert mock_instance.models.generate_content.call_count == 2

