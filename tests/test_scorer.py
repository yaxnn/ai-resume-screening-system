"""
Tests for  scoring logic.
"""

import pytest
from src.models import Candidate, ParseStatus
from src.eligibility import EligibilityResult
from src.scorer import (
    score_ai_project_depth,
    score_python_backend,
    score_cloud_fullstack,
    score_engineering_depth,
    score_candidate,
    score_and_rank_candidates,
)
from unittest.mock import patch

def test_score_ai_project_depth_max_40():
    text = "RAG vector search LangGraph embeddings tool calling multi-agent RAGAS evaluation state management LangChain OpenAI API LlamaIndex Hugging Face fine-tuning vLLM LLM"
    score, strg, conc, evid = score_ai_project_depth(text)
    assert score == 40
    assert len(strg) > 0
    assert len(evid) > 0

def test_score_python_backend_max_30():
    text = "Python FastAPI PostgreSQL Redis Celery async architecture data processing Flask Django SQLAlchemy"
    score, _, _, _ = score_python_backend(text)
    assert score == 30

def test_score_cloud_fullstack_max_15():
    text = "GCP AWS Docker Kubernetes React Next.js deployment end-to-end full stack"
    score, _, _, _ = score_cloud_fullstack(text)
    assert score == 15

def test_score_engineering_depth_max_5():
    text = "testing pytest CI/CD caching queues observability concurrency failure handling reliability architecture"
    score, _, _, _ = score_engineering_depth(text)
    assert score == 5

def test_shallow_ai_project_penalty():
    text = "built a wrapper simple API call basic script LangChain"
    score, strg, conc, evid = score_ai_project_depth(text)
    assert score < 3  # LangChain is 3, penalty is 5 -> clamped to 0
    assert score == 0
    assert any("shallow API wrapper" in c for c in conc)

def test_tutorial_project_penalty():
    text = "RAG LangChain bootcamp project tutorial udemy course"
    score, strg, conc, evid = score_ai_project_depth(text)
    # RAG (5) + LangChain (3) = 8. Penalty = -10. -> 0
    assert score == 0
    assert any("tutorial" in c for c in conc)

@patch("src.github_client.fetch_github_data")
def test_github_score_is_zero(mock_fetch):
    from src.github_client import GithubData
    mock_fetch.return_value = GithubData(username="user", valid_profile=False, error="API rate limit exceeded")
    cand = Candidate(
        source_file="dummy.pdf",
        extracted_text="Python RAG FastAPI GCP testing",
        skills=["Python"],
        projects=["RAG System"],
        github_url="https://github.com/user"
    )
    elig = EligibilityResult(eligible=True)
    res = score_candidate(cand, elig)
    assert res.breakdown.github == 0
    assert "unavailable" in res.github_enrichment_status

def test_rejected_candidates_not_ranked():
    cand1 = Candidate(source_file="f1.pdf", extracted_text="Python RAG")
    elig1 = EligibilityResult(eligible=True)
    
    cand2 = Candidate(source_file="f2.pdf", extracted_text="")
    elig2 = EligibilityResult(eligible=False)
    
    cand3 = Candidate(source_file="f3.pdf", extracted_text="Python RAG AWS")
    elig3 = EligibilityResult(eligible=True)

    results = score_and_rank_candidates([
        (cand1, elig1),
        (cand2, elig2),
        (cand3, elig3)
    ])
    
    r2 = next(r for c, e, r in results if c.source_file == "f2.pdf")
    assert r2.rank is None
    assert r2.total_score == 0
    
    # Check ranking order
    r1 = next(r for c, e, r in results if c.source_file == "f1.pdf")
    r3 = next(r for c, e, r in results if c.source_file == "f3.pdf")
    
    # r3 has AWS (+3), r1 doesn't.
    assert r3.rank == 1
    assert r1.rank == 2

def test_framework_only_in_skills_penalty():
    cand = Candidate(
        source_file="f.pdf",
        extracted_text="Skills: Python, LangChain, RAG, vector search, tool calling, embeddings, RAGAS, OpenAI API",
        skills=["Python", "LangChain", "RAG", "vector search"],
        projects=[] # Empty projects
    )
    elig = EligibilityResult(eligible=True)
    res = score_candidate(cand, elig)
    
    # Should get the penalty
    assert any("appear without any project descriptions" in c for c in res.concerns)
    # Total score should have been reduced by 10
    # AI score = 40 (based on keywords), Python = 10, total = 50. Penalty -10 => 40.
    assert res.breakdown.ai_project_depth <= 30


from unittest.mock import patch
from src.llm_evaluator import LLMEvaluation

@patch("src.llm_evaluator.evaluate_candidate_llm")
def test_score_candidate_llm_success(mock_eval):
    mock_eval.return_value = LLMEvaluation(
        ai_project_score=30,
        python_backend_score=25,
        cloud_fullstack_score=10,
        engineering_depth_score=4,
        strengths=["Good AI"],
        concerns=["No tests"],
        project_summary="Built things",
        evidence_citations=["Snippet"],
        confidence_level="High"
    )
    
    cand = Candidate(source_file="f.pdf", extracted_text="Python RAG AWS")
    elig = EligibilityResult(eligible=True)
    
    # Force use LLM
    res = score_candidate(cand, elig, use_llm=True)
    
    assert res.scoring_method == "llm_enriched"
    assert res.llm_evaluation_status == "success"
    assert res.breakdown.ai_project_depth == 30
    assert res.total_score == 69  # 30+25+10+0+4
    assert res.project_summary == "Built things"

@patch("src.llm_evaluator.evaluate_candidate_llm")
def test_score_candidate_llm_fallback(mock_eval):
    mock_eval.return_value = None
    
    cand = Candidate(source_file="f.pdf", extracted_text="Python RAG AWS")
    elig = EligibilityResult(eligible=True)
    
    res = score_candidate(cand, elig, use_llm=True)
    
    assert res.scoring_method == "deterministic_rule_based"
    assert res.llm_evaluation_status == "failed_fallback_to_rules"
    assert res.project_summary is None


@patch("src.github_client.fetch_github_data")
def test_score_github_profile_success(mock_fetch):
    from src.github_client import GithubData
    
    mock_fetch.return_value = GithubData(
        username="testuser",
        valid_profile=True,
        error=None,
        repos=[
            {"name": "test-repo", "language": "Python"},
            {"name": "ai-project", "language": "Jupyter Notebook", "description": "Langchain llm"}
        ],
        events=[
            {"type": "PushEvent"},
            {"type": "PushEvent"},
            {"type": "PullRequestEvent"},
        ]
    )
    
    cand = Candidate(
        source_file="f.pdf",
        extracted_text="Python RAG AWS",
        github_url="https://github.com/testuser"
    )
    elig = EligibilityResult(eligible=True)
    
    res = score_candidate(cand, elig)
    
    # 3 events -> (3//2)+1 = 2 pts for activity
    # 2 relevant repos -> 2*2 = 4 pts for repos
    # total github score = 6
    assert res.breakdown.github == 6
    assert res.github_enrichment_status == "success"
    assert res.github_profile == "testuser"
    assert "3 recent meaningful events" in res.github_activity_summary
    assert "2 relevant" in res.github_repo_summary

@patch("src.github_client.fetch_github_data")
def test_score_github_profile_max_10(mock_fetch):
    from src.github_client import GithubData
    
    # 20 events, 10 relevant repos
    mock_fetch.return_value = GithubData(
        username="testuser",
        valid_profile=True,
        repos=[{"name": f"repo{i}", "language": "Python"} for i in range(10)],
        events=[{"type": "PushEvent"} for i in range(20)]
    )
    
    from src.scorer import score_github_profile
    score, status, profile, act_sum, repo_sum, strg, conc, evid = score_github_profile("https://github.com/testuser")
    
    assert score == 10  # Capped at 10 (5 for activity, 5 for repos)
    assert status == "success"

def test_score_github_profile_invalid_url():
    from src.scorer import score_github_profile
    score, status, profile, act_sum, repo_sum, strg, conc, evid = score_github_profile("not a url")
    
    assert score == 0
    assert "unavailable" in status

@patch("src.github_client.fetch_github_data")
def test_score_github_profile_api_error(mock_fetch):
    from src.github_client import GithubData
    
    mock_fetch.return_value = GithubData(
        username="testuser",
        valid_profile=False,
        error="API rate limit exceeded"
    )
    
    from src.scorer import score_github_profile
    score, status, profile, act_sum, repo_sum, strg, conc, evid = score_github_profile("https://github.com/testuser")
    
    assert score == 0
    assert status == "unavailable (API rate limit exceeded)"
    assert profile == "testuser"
