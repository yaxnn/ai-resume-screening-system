"""
llm_evaluator.py — LLM Enrichment for AI Resume Screening System .

Uses Google Gen AI SDK to semantically evaluate candidate project depth,
augmenting the deterministic rule-based scoring baseline.
"""

import os
import json
import logging
import time
from typing import Optional, List
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

class LLMEvaluation(BaseModel):
    """Structured response schema for LLM candidate evaluation."""
    ai_project_score: int = Field(ge=0, le=40, description="Score for AI/Agentic/RAG project depth (0-40). Deduct for shallow API wrappers or tutorials.")
    python_backend_score: int = Field(ge=0, le=30, description="Score for Python and backend engineering (0-30).")
    cloud_fullstack_score: int = Field(ge=0, le=15, description="Score for Cloud, deployment, and full stack (0-15).")
    engineering_depth_score: int = Field(ge=0, le=5, description="Score for engineering depth and reliability (0-5).")
    strengths: List[str] = Field(description="Evidence-backed strengths found in the resume.")
    concerns: List[str] = Field(description="Concerns such as thin implementations or lack of detail.")
    project_summary: str = Field(description="A concise summary of the candidate's technical projects.")
    evidence_citations: List[str] = Field(description="Short evidence excerpts cited from the extracted resume text supporting the scores.")
    confidence_level: str = Field(description="High, Medium, or Low confidence in the evaluation based on resume clarity.")

def evaluate_candidate_llm(candidate_text: str, max_retries: int = 2) -> Optional[LLMEvaluation]:
    """
    Evaluates candidate text using the configured LLM and returns structured scoring data.
    Returns None if the evaluation fails or API key is missing.
    """
    if genai is None:
        logger.warning("google-genai package not installed. Skipping LLM evaluation.")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.debug("GEMINI_API_KEY not set. Skipping LLM evaluation.")
        return None

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are an expert technical recruiter and senior engineer evaluating an SDE intern candidate.
Based on the following extracted resume text, evaluate the candidate's engineering experience.

IMPORTANT RULES:
1. Use only the provided text. Do NOT invent projects, technologies, or work experience.
2. Distinguish substantive implementations from thin LLM/API wrappers and tutorial-style projects. 
   Penalize tutorials or simple API wrappers by giving lower scores in the AI category.
3. Keep citations short and verbatim from the text.
4. Provide integer scores for the 4 categories within their respective maximum limits.

Resume Text:
---
{candidate_text}
---
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LLMEvaluation,
                    temperature=0.1
                )
            )
            if not response.text:
                raise ValueError("Empty response from LLM")
                
            data = json.loads(response.text)
            return LLMEvaluation(**data)
            
        except Exception as e:
            logger.warning(f"LLM evaluation failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                logger.error("Max retries reached for LLM evaluation.")
                return None
    
    return None

