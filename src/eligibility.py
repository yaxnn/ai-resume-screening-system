"""
eligibility.py — Deterministic eligibility filtering for SDE internship .

Checks two mandatory criteria:
    1. **Python evidence** — the candidate demonstrates Python usage in skills,
       projects, work experience, or implementation descriptions.
    2. **Meaningful AI/LLM/RAG/Agentic evidence** — the candidate has worked on
       or built something substantive involving LLMs, RAG, embeddings, agentic
       frameworks, or equivalent.

Both conditions must be met for eligibility.  The module uses case-insensitive
matching with word boundaries to avoid false positives (e.g. "storage" ≠ "RAG").

This module is **deterministic** — no LLM or external API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from src.models import Candidate, ParseStatus

# ---------------------------------------------------------------------------
# Configurable keyword lists
# ---------------------------------------------------------------------------

# Python evidence keywords — matched with word boundaries, case-insensitive
PYTHON_KEYWORDS: List[str] = [
    "Python",
    "Django",
    "Flask",
    "FastAPI",
    "Streamlit",
    "Jupyter",
    "Pandas",
    "NumPy",
    "SciPy",
    "scikit-learn",
    "sklearn",
    "PyTorch",
    "TensorFlow",
    "Keras",
    "Matplotlib",
    "Seaborn",
    "Celery",
    "Scrapy",
    "BeautifulSoup",
    "Uvicorn",
    "Gunicorn",
    "Pydantic",
    "SQLAlchemy",
    "Pytest",
    "pip",
]

# AI/LLM framework and concept keywords — these indicate the *type* of work
# but a bare mention is not necessarily proof of a meaningful project.
# Divided into two tiers:
#   - STRONG: Frameworks/tools that almost always imply hands-on AI project work.
#   - CONTEXTUAL: Concepts that need supporting evidence (a project description,
#     an implementation detail, or co-occurrence with a strong keyword).

AI_STRONG_KEYWORDS: List[str] = [
    # Agentic / orchestration frameworks
    "LangChain",
    "LangGraph",
    "LlamaIndex",
    "Llama_Index",
    "Google ADK",
    "CrewAI",
    "AutoGen",
    "Haystack",
    "Semantic Kernel",
    "Agno",
    # RAG-specific
    "RAG pipeline",
    "RAG system",
    "RAG architecture",
    "RAG application",
    "RAG-based",
    "Retrieval-Augmented",
    "retrieval augmented generation",
    # Vector / embedding infra
    "vector database",
    "vector store",
    "vector search",
    "Pinecone",
    "Weaviate",
    "Chroma",
    "ChromaDB",
    "Milvus",
    "Qdrant",
    "FAISS",
    "pgvector",
    # LLM APIs & models
    "OpenAI API",
    "GPT-3",
    "GPT-4",
    "GPT-3.5",
    "Claude API",
    "Anthropic API",
    "Gemini API",
    "Groq API",
    "Ollama",
    "vLLM",
    "Hugging Face",
    "HuggingFace",
    "fine-tuning",
    "fine-tuned",
    "finetuning",
    "finetuned",
    # Evaluation
    "RAGAS",
    "LLM evaluation",
    "prompt engineering",
    # Multi-agent
    "multi-agent",
    "multi agent",
    "tool-calling",
    "tool calling",
    "function calling",
    "agentic workflow",
    "agentic system",
    "agentic AI",
    "AI agent",
]

AI_CONTEXTUAL_KEYWORDS: List[str] = [
    # These need at least one co-occurring strong keyword OR a project context
    "LLM",
    "RAG",
    "embedding",
    "embeddings",
    "vector",
    "agent",
    "transformer",
    "BERT",
    "GPT",
    "NLP",
    "chatbot",
    "generative AI",
    "GenAI",
    "Gen AI",
]

# Phrases that should NOT count as meaningful AI project evidence on their own
# (e.g. "AI/ML" in a skills list with no supporting project).
AI_NOISE_PHRASES: List[str] = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "AI/ML",
    "ML/AI",
    "AI & ML",
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EligibilityResult:
    """Outcome of the eligibility check for one candidate.

    Attributes:
        eligible: Whether the candidate meets both criteria.
        rejection_reasons: Human-readable list of reasons for rejection.
        matched_skills: Skills from the candidate's skills list that are
            relevant to the position.
        python_evidence: Snippets or references showing Python usage.
        ai_evidence: Snippets or references showing AI/LLM project work.
    """

    eligible: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    matched_skills: List[str] = field(default_factory=list)
    python_evidence: List[str] = field(default_factory=list)
    ai_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary for JSON export."""
        return {
            "eligible": self.eligible,
            "rejection_reasons": self.rejection_reasons,
            "matched_skills": self.matched_skills,
            "python_evidence": self.python_evidence,
            "ai_evidence": self.ai_evidence,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_word_pattern(keyword: str) -> re.Pattern:
    """Compile a case-insensitive word-boundary pattern for *keyword*.

    Handles multi-word phrases, hyphens, underscores, and special characters
    (e.g. ``scikit-learn``, ``GPT-3.5``, ``AI/ML``) by escaping the keyword
    and using ``\\b`` boundaries.
    """
    escaped = re.escape(keyword)
    # Allow flexible whitespace/hyphens in multi-word phrases
    # e.g. "RAG pipeline" should match "RAG  pipeline" or "RAG-pipeline"
    escaped = re.sub(r"\\ ", r"[\\s\\-]+", escaped)
    return re.compile(rf"\b{escaped}\b", re.IGNORECASE)


def _find_evidence(
    text: str,
    keywords: Sequence[str],
    *,
    max_snippets: int = 5,
    context_chars: int = 80,
) -> List[str]:
    """Search *text* for each keyword and return contextual snippets.

    Parameters
    ----------
    text:
        The full text to search.
    keywords:
        Keywords / phrases to look for.
    max_snippets:
        Maximum number of evidence snippets to collect.
    context_chars:
        How many characters of surrounding context to include.

    Returns
    -------
    list[str]
        Deduplicated evidence snippets.
    """
    snippets: List[str] = []
    seen_keywords: set = set()

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in seen_keywords:
            continue

        pattern = _build_word_pattern(kw)
        match = pattern.search(text)
        if match:
            seen_keywords.add(kw_lower)
            start = max(0, match.start() - context_chars)
            end = min(len(text), match.end() + context_chars)
            # Extract context and clean up whitespace
            snippet = text[start:end].strip()
            snippet = re.sub(r"\s+", " ", snippet)
            snippets.append(f"[{kw}] ...{snippet}...")
            if len(snippets) >= max_snippets:
                break

    return snippets


def _has_keyword_match(text: str, keywords: Sequence[str]) -> bool:
    """Return True if any keyword matches in *text* with word boundaries."""
    for kw in keywords:
        pattern = _build_word_pattern(kw)
        if pattern.search(text):
            return True
    return False


def _matched_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    """Return all keywords that match in *text* with word boundaries."""
    matched: List[str] = []
    for kw in keywords:
        pattern = _build_word_pattern(kw)
        if pattern.search(text):
            matched.append(kw)
    return matched


# ---------------------------------------------------------------------------
# Core eligibility function
# ---------------------------------------------------------------------------


def check_eligibility(candidate: Candidate) -> EligibilityResult:
    """Determine whether *candidate* is eligible for the SDE internship.

    Eligibility requires **both**:
        1. Python evidence (skills, projects, or full-text mentions).
        2. Meaningful AI/LLM/RAG/agentic project evidence.

    The function searches:
        - ``candidate.skills`` (structured list)
        - ``candidate.projects`` (structured list)
        - ``candidate.extracted_text`` (full resume text, catching
          experience sections, summaries, etc.)

    Parameters
    ----------
    candidate:
        A parsed Candidate record from .

    Returns
    -------
    EligibilityResult
        Populated result with evidence and/or rejection reasons.
    """
    result = EligibilityResult()

    # ------------------------------------------------------------------ #
    #  Guard: unparsed candidates are automatically ineligible             #
    # ------------------------------------------------------------------ #
    if candidate.parse_status != ParseStatus.SUCCESS:
        result.rejection_reasons.append(
            f"Resume could not be parsed: {candidate.parse_error or 'unknown error'}"
        )
        return result

    full_text = candidate.extracted_text or ""

    # Combine structured fields into searchable text blocks
    skills_text = " ".join(candidate.skills)
    projects_text = " ".join(candidate.projects)
    combined_text = f"{skills_text} {projects_text} {full_text}"

    # ------------------------------------------------------------------ #
    #  1. Python evidence                                                  #
    # ------------------------------------------------------------------ #
    python_evidence = _find_evidence(combined_text, PYTHON_KEYWORDS)
    has_python = len(python_evidence) > 0

    if has_python:
        result.python_evidence = python_evidence
    else:
        result.rejection_reasons.append(
            "No Python evidence found in skills, projects, or resume text."
        )

    # ------------------------------------------------------------------ #
    #  2. AI / LLM / RAG / Agentic evidence                               #
    # ------------------------------------------------------------------ #
    # First look for strong indicators (frameworks, specific tools)
    strong_matches = _matched_keywords(combined_text, AI_STRONG_KEYWORDS)
    strong_evidence = _find_evidence(combined_text, AI_STRONG_KEYWORDS)

    # Then look for contextual keywords
    contextual_matches = _matched_keywords(combined_text, AI_CONTEXTUAL_KEYWORDS)

    has_meaningful_ai = False

    if strong_matches:
        # Strong keywords are sufficient on their own
        has_meaningful_ai = True
        result.ai_evidence = strong_evidence
    elif contextual_matches:
        # Contextual keywords need supporting context: look for them inside
        # project descriptions, experience blocks, or near action verbs
        # that suggest implementation work.
        project_context_patterns = [
            r"(?:built|developed|implemented|designed|created|deployed|"
            r"architected|integrated|engineered|trained|optimized|constructed)"
            r"[^.]{0,120}",
        ]
        for ctx_kw in contextual_matches:
            ctx_pattern = _build_word_pattern(ctx_kw)
            for proj_pattern_str in project_context_patterns:
                proj_re = re.compile(proj_pattern_str, re.IGNORECASE)
                for m in proj_re.finditer(combined_text):
                    block = m.group(0)
                    if ctx_pattern.search(block):
                        has_meaningful_ai = True
                        snippet = re.sub(r"\s+", " ", block.strip())[:150]
                        result.ai_evidence.append(f"[{ctx_kw}] ...{snippet}...")
                        break
                if has_meaningful_ai:
                    break
            if has_meaningful_ai:
                break

        # Fallback: if multiple contextual keywords co-occur, that's also
        # reasonably strong evidence (e.g. "LLM" + "RAG" + "embedding")
        if not has_meaningful_ai and len(contextual_matches) >= 2:
            has_meaningful_ai = True
            result.ai_evidence = _find_evidence(
                combined_text, contextual_matches, max_snippets=3
            )

    if not has_meaningful_ai:
        result.rejection_reasons.append(
            "No meaningful AI/LLM/RAG/agentic project evidence found."
        )

    # ------------------------------------------------------------------ #
    #  3. Collect matched skills (for reporting)                           #
    # ------------------------------------------------------------------ #
    all_relevant_kws = PYTHON_KEYWORDS + AI_STRONG_KEYWORDS + AI_CONTEXTUAL_KEYWORDS
    result.matched_skills = _matched_keywords(combined_text, all_relevant_kws)

    # ------------------------------------------------------------------ #
    #  4. Final verdict                                                    #
    # ------------------------------------------------------------------ #
    result.eligible = has_python and has_meaningful_ai

    return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def check_all_eligibility(
    candidates: List[Candidate],
) -> List[Dict]:
    """Run eligibility checks on a list of candidates.

    Parameters
    ----------
    candidates:
        Parsed Candidate records from .

    Returns
    -------
    list[dict]
        One dictionary per candidate containing candidate metadata and the
        eligibility result.
    """
    results: List[Dict] = []

    for candidate in candidates:
        elig = check_eligibility(candidate)
        results.append(
            {
                "source_file": candidate.source_file,
                "filename": candidate.filename,
                "candidate_name": candidate.candidate_name,
                "email": candidate.email,
                "github_url": candidate.github_url,
                "parse_status": candidate.parse_status.value,
                "eligibility": elig.to_dict(),
            }
        )

    return results

