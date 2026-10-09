"""
scorer.py — Deterministic rule-based scoring engine .

Calculates a 100-point score for eligible candidates based on:
- AI/RAG Project Depth (40 pts)
- Python & Backend Engineering (30 pts)
- Cloud/Full Stack (15 pts)
- GitHub Activity (10 pts - pending)
- Engineering Depth (5 pts)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models import Candidate
from src.eligibility import EligibilityResult


@dataclass
class ScoreBreakdown:
    ai_project_depth: int = 0
    python_backend: int = 0
    cloud_fullstack: int = 0
    github: int = 0
    engineering_depth: int = 0


@dataclass
class ScoringResult:
    total_score: int = 0
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    rank: Optional[int] = None
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    scoring_evidence: List[str] = field(default_factory=list)
    scoring_method: str = "deterministic_rule_based"
    github_enrichment_status: str = "pending"
    project_summary: Optional[str] = None
    llm_evaluation_status: str = "disabled"
    
    # : GitHub Summaries
    github_profile: Optional[str] = None
    github_activity_summary: Optional[str] = None
    github_repo_summary: Optional[str] = None
    category_explanations: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_score": self.total_score,
            "breakdown": {
                "ai_project_depth": self.breakdown.ai_project_depth,
                "python_backend": self.breakdown.python_backend,
                "cloud_fullstack": self.breakdown.cloud_fullstack,
                "github": self.breakdown.github,
                "engineering_depth": self.breakdown.engineering_depth,
            },
            "rank": self.rank,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "scoring_evidence": self.scoring_evidence,
            "scoring_method": self.scoring_method,
            "github_enrichment_status": self.github_enrichment_status,
            "project_summary": self.project_summary,
            "llm_evaluation_status": self.llm_evaluation_status,
            "github_profile": self.github_profile,
            "github_activity_summary": self.github_activity_summary,
            "github_repo_summary": self.github_repo_summary,
            "category_explanations": self.category_explanations,
        }



# ---------------------------------------------------------------------------
# Scoring Keywords & Rules
# ---------------------------------------------------------------------------

def _count_matches(text: str, keywords: List[str]) -> int:
    """Return how many distinct keywords match in the text."""
    count = 0
    for kw in keywords:
        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        if pattern.search(text):
            count += 1
    return count

def _has_match(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword matches in the text."""
    return _count_matches(text, keywords) > 0


def score_ai_project_depth(text: str) -> Tuple[int, List[str], List[str], List[str]]:
    """Score out of 40 points."""
    score = 0
    strengths = []
    concerns = []
    evidence = []

    # High value concepts (5 pts each)
    advanced_concepts = [
        "RAG", "retrieval augmented generation", "vector search", "vector database",
        "Pinecone", "Chroma", "Weaviate", "embeddings", "tool calling", "agentic workflow",
        "multi-agent", "evaluation", "RAGAS", "LangGraph", "state management", "orchestration",
        "Semantic Kernel", "AutoGen", "CrewAI", "Agno"
    ]
    adv_count = _count_matches(text, advanced_concepts)
    if adv_count > 0:
        pts = min(25, adv_count * 5)
        score += pts
        strengths.append(f"Advanced AI concepts found ({pts} pts)")
        evidence.append(f"Matched advanced AI terms: {adv_count}")

    # Standard AI frameworks (3 pts each)
    standard_frameworks = [
        "LangChain", "LlamaIndex", "OpenAI API", "Hugging Face", "fine-tuning", "vLLM", "LLM",
        "Claude API", "Anthropic API", "Gemini API", "Groq API", "Ollama"
    ]
    std_count = _count_matches(text, standard_frameworks)
    if std_count > 0:
        pts = min(15, std_count * 3)
        score += pts
        strengths.append(f"AI frameworks/APIs found ({pts} pts)")
        evidence.append(f"Matched standard AI frameworks: {std_count}")

    # Penalties for shallow or tutorial
    tutorial_keywords = ["tutorial", "bootcamp", "coursework", "udemy", "coursera", "class project"]
    if _has_match(text, tutorial_keywords):
        score -= 10
        concerns.append("Project may be from a tutorial or coursework (-10 pts)")
        evidence.append("Found tutorial/coursework keywords")

    wrapper_keywords = ["wrapper", "simple api call", "basic script"]
    if _has_match(text, wrapper_keywords) and adv_count == 0:
        score -= 5
        concerns.append("AI project appears to be a shallow API wrapper (-5 pts)")
        evidence.append("Found shallow wrapper keywords without advanced concepts")
    
    # Floor at 0, cap at 40
    score = max(0, min(40, score))
    return score, strengths, concerns, evidence


def score_python_backend(text: str) -> Tuple[int, List[str], List[str], List[str]]:
    """Score out of 30 points."""
    score = 0
    strengths = []
    concerns = []
    evidence = []

    backend_kws = ["FastAPI", "Django", "Flask", "async", "Celery", "PostgreSQL", "Redis", "SQLAlchemy", "architecture", "data processing"]
    count = _count_matches(text, backend_kws)
    
    if count > 0:
        pts = min(20, count * 5)
        score += pts
        strengths.append(f"Python backend engineering tools found ({pts} pts)")
        evidence.append(f"Matched backend keywords: {count}")
    
    if _has_match(text, ["Python"]):
        score += 10
        strengths.append("Python experience found (10 pts)")
        evidence.append("Matched Python")
    elif count > 0:
        concerns.append("Backend tools mentioned, but missing direct 'Python' keyword")
    else:
        concerns.append("No strong Python or backend engineering evidence found")
        
    score = max(0, min(30, score))
    return score, strengths, concerns, evidence


def score_cloud_fullstack(text: str) -> Tuple[int, List[str], List[str], List[str]]:
    """Score out of 15 points."""
    score = 0
    strengths = []
    concerns = []
    evidence = []

    cloud_kws = ["GCP", "AWS", "Azure", "Docker", "Kubernetes", "deployment", "end-to-end", "React", "Next.js", "frontend", "full stack"]
    count = _count_matches(text, cloud_kws)
    
    if count > 0:
        pts = min(15, count * 3)
        score += pts
        strengths.append(f"Cloud/Fullstack experience found ({pts} pts)")
        evidence.append(f"Matched cloud/fullstack keywords: {count}")
    else:
        concerns.append("No cloud, deployment, or full stack evidence found")
        
    score = max(0, min(15, score))
    return score, strengths, concerns, evidence


def score_engineering_depth(text: str) -> Tuple[int, List[str], List[str], List[str]]:
    """Score out of 5 points."""
    score = 0
    strengths = []
    concerns = []
    evidence = []

    depth_kws = ["testing", "pytest", "CI/CD", "caching", "queues", "observability", "concurrency", "failure handling", "reliability", "architecture"]
    count = _count_matches(text, depth_kws)
    
    if count > 0:
        pts = min(5, count * 2)
        score += pts
        strengths.append(f"Engineering depth signals found ({pts} pts)")
        evidence.append(f"Matched engineering depth keywords: {count}")
    else:
        concerns.append("No specific engineering depth signals (testing, CI/CD, etc.) found")
        
    score = max(0, min(5, score))
    return score, strengths, concerns, evidence


def score_github_profile(github_url: Optional[str]) -> Tuple[int, str, Optional[str], Optional[str], List[str], List[str], List[str]]:
    """
    Score GitHub out of 10 points:
    - Recent activity: 0-5 pts
    - Maintained/relevant repos: 0-5 pts
    
    Returns:
        score, status, profile, activity_summary, repo_summary, strengths, concerns, evidence
    """
    if not github_url:
        return 0, "unavailable (no url)", None, None, None, [], [], []
        
    try:
        from src.github_client import fetch_github_data
    except ImportError:
        return 0, "unavailable (import error)", None, None, None, [], [], []

    data = fetch_github_data(github_url)
    if not data.username:
        return 0, "unavailable (invalid url)", None, None, None, [], [], []
        
    if not data.valid_profile:
        err = data.error or "unknown error"
        return 0, f"unavailable ({err})", data.username, None, None, [], [], []

    score = 0
    strengths = []
    concerns = []
    evidence = []
    
    # 1. Activity (max 5)
    activity_score = 0
    meaningful_events = [e for e in data.events if e.get("type") in ("PushEvent", "CreateEvent", "PullRequestEvent")]
    num_events = len(meaningful_events)
    
    if num_events > 0:
        activity_score = min(5, (num_events // 2) + 1)
        score += activity_score
        strengths.append(f"Recent GitHub activity found ({activity_score} pts)")
        evidence.append(f"Found {num_events} meaningful public events.")
    else:
        concerns.append("No recent meaningful public GitHub activity.")
        
    activity_summary = f"{num_events} recent meaningful events"
    
    # 2. Repositories (max 5)
    repo_score = 0
    ai_kws = ["llm", "rag", "agent", "ai", "machine learning", "deep learning", "langchain", "pytorch", "tensorflow"]
    backend_kws = ["python", "django", "fastapi", "flask", "backend", "api"]
    
    relevant_repos = 0
    for repo in data.repos:
        name_desc = f"{repo.get('name') or ''} {repo.get('description') or ''}".lower()
        lang = (repo.get("language") or "").lower()
        
        is_relevant = False
        if lang == "python":
            is_relevant = True
        if any(kw in name_desc for kw in ai_kws + backend_kws):
            is_relevant = True
            
        if is_relevant:
            relevant_repos += 1

    if relevant_repos > 0:
        repo_score = min(5, relevant_repos * 2)
        score += repo_score
        strengths.append(f"Relevant public repos found ({repo_score} pts)")
        evidence.append(f"Found {relevant_repos} relevant public repos (Python/Backend/AI).")
    else:
        concerns.append("No relevant public repos identified.")
        
    repo_summary = f"{len(data.repos)} public repos ({relevant_repos} relevant)"
    
    return score, "success", data.username, activity_summary, repo_summary, strengths, concerns, evidence

def score_candidate(candidate: Candidate, elig_result: EligibilityResult, use_llm: bool = False) -> ScoringResult:
    """Score a candidate if they are eligible."""
    if not elig_result.eligible:
        return ScoringResult()

    full_text = f"{' '.join(candidate.skills)} {' '.join(candidate.projects)} {candidate.extracted_text or ''}"

    # Rule-based calculation first
    s_ai, str_ai, con_ai, ev_ai = score_ai_project_depth(full_text)
    s_py, str_py, con_py, ev_py = score_python_backend(full_text)
    s_cl, str_cl, con_cl, ev_cl = score_cloud_fullstack(full_text)
    s_en, str_en, con_en, ev_en = score_engineering_depth(full_text)
    
    # GitHub calculation
    gh_score, gh_status, gh_profile, gh_act_sum, gh_repo_sum, str_gh, con_gh, ev_gh = score_github_profile(candidate.github_url)
    
    breakdown = ScoreBreakdown(
        ai_project_depth=s_ai,
        python_backend=s_py,
        cloud_fullstack=s_cl,
        github=gh_score,
        engineering_depth=s_en
    )
    
    total = s_ai + s_py + s_cl + gh_score + s_en
    
    strengths = str_ai + str_py + str_cl + str_en + str_gh
    concerns = con_ai + con_py + con_cl + con_en + con_gh
    evidence = ev_ai + ev_py + ev_cl + ev_en + ev_gh
    
    # Penalize framework names appearing ONLY in skills list without supporting project info.
    if s_ai > 10 and not candidate.projects:
        total -= 10
        total = max(0, total)
        breakdown.ai_project_depth = max(0, breakdown.ai_project_depth - 10)
        concerns.append("AI framework names appear without any project descriptions (-10 pts)")
        evidence.append("No projects found in parsed data despite AI keywords")
        con_ai.append("AI framework names appear without any project descriptions (-10 pts)")
        ev_ai.append("No projects found in parsed data despite AI keywords")

    category_explanations = {
        "ai_project_depth": {"strengths": str_ai, "concerns": con_ai, "evidence": ev_ai},
        "python_backend": {"strengths": str_py, "concerns": con_py, "evidence": ev_py},
        "cloud_fullstack": {"strengths": str_cl, "concerns": con_cl, "evidence": ev_cl},
        "github": {"strengths": str_gh, "concerns": con_gh, "evidence": ev_gh},
        "engineering_depth": {"strengths": str_en, "concerns": con_en, "evidence": ev_en},
    }

    result = ScoringResult(
        total_score=total,
        breakdown=breakdown,
        strengths=strengths,
        concerns=concerns,
        scoring_evidence=evidence,
        scoring_method="deterministic_rule_based",
        llm_evaluation_status="disabled",
        github_enrichment_status=gh_status,
        github_profile=gh_profile,
        github_activity_summary=gh_act_sum,
        github_repo_summary=gh_repo_sum,
        category_explanations=category_explanations,
    )

    if use_llm:
        from src.llm_evaluator import evaluate_candidate_llm
        llm_eval = evaluate_candidate_llm(full_text)
        
        if llm_eval:
            # Override with LLM scores, clamped to valid ranges
            result.breakdown.ai_project_depth = max(0, min(40, llm_eval.ai_project_score))
            result.breakdown.python_backend = max(0, min(30, llm_eval.python_backend_score))
            result.breakdown.cloud_fullstack = max(0, min(15, llm_eval.cloud_fullstack_score))
            result.breakdown.engineering_depth = max(0, min(5, llm_eval.engineering_depth_score))
            
            result.total_score = (
                result.breakdown.ai_project_depth + 
                result.breakdown.python_backend + 
                result.breakdown.cloud_fullstack + 
                result.breakdown.github + 
                result.breakdown.engineering_depth
            )
            
            result.strengths = llm_eval.strengths
            result.concerns = llm_eval.concerns
            result.scoring_evidence = llm_eval.evidence_citations
            result.project_summary = llm_eval.project_summary
            
            result.scoring_method = "llm_enriched"
            result.llm_evaluation_status = "success"
        else:
            result.llm_evaluation_status = "failed_fallback_to_rules"
            
    return result


def score_and_rank_candidates(
    candidates_with_eligibility: List[Tuple[Candidate, EligibilityResult]],
    use_llm: bool = False
) -> List[Tuple[Candidate, EligibilityResult, ScoringResult]]:
    """Score all candidates and rank the eligible ones."""
    
    results = []
    for cand, elig in candidates_with_eligibility:
        scoring = score_candidate(cand, elig, use_llm=use_llm)
        results.append((cand, elig, scoring))
        
    # Filter and sort eligible for ranking
    eligible_results = [r for r in results if r[1].eligible]
    eligible_results.sort(key=lambda r: r[2].total_score, reverse=True)
    
    # Assign ranks
    rank = 1
    for r in eligible_results:
        r[2].rank = rank
        rank += 1
        
    return results

