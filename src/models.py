"""
models.py — Pydantic data models for the AI Resume Screening System .

Defines the Candidate model that represents all extracted information for a
single resume.  No scoring or LLM fields are included here; those belong to
other modules.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ParseStatus(str, Enum):
    """Possible outcomes for resume parsing."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class Candidate(BaseModel):
    """Structured representation of a parsed resume.

    All extraction fields are *conservative*: they are populated only when
    the relevant information is clearly present in the resume text.  Missing
    values are represented as ``None`` or empty lists — never invented.

    Attributes:
        candidate_name: Full name extracted from the resume, or ``None``.
        email: Primary e-mail address found in the resume, or ``None``.
        skills: Deduplicated list of skills mentioned under a skills section.
        projects: List of project titles or descriptions found in the resume.
        github_url: GitHub profile URL extracted from the resume, or ``None``.
        source_file: Absolute path of the file that was parsed.
        extracted_text: Full plain-text content extracted from the file.
        parse_status: Whether parsing succeeded, failed, or was skipped.
        parse_error: Human-readable error message when ``parse_status`` is
            ``FAILURE``; ``None`` otherwise.
    """

    candidate_name: Optional[str] = Field(
        default=None,
        description="Full name of the candidate as it appears in the resume.",
    )
    email: Optional[str] = Field(
        default=None,
        description="Primary e-mail address found in the resume.",
    )
    skills: List[str] = Field(
        default_factory=list,
        description="Deduplicated list of skills extracted from the resume.",
    )
    projects: List[str] = Field(
        default_factory=list,
        description="Project titles or short descriptions found in the resume.",
    )
    github_url: Optional[str] = Field(
        default=None,
        description="GitHub profile URL found in the resume.",
    )
    source_file: str = Field(
        description="Absolute path (as a string) of the source resume file.",
    )
    extracted_text: str = Field(
        default="",
        description="Full plain-text content extracted from the resume.",
    )
    parse_status: ParseStatus = Field(
        default=ParseStatus.SUCCESS,
        description="Outcome of the parsing attempt.",
    )
    parse_error: Optional[str] = Field(
        default=None,
        description="Error description when parse_status is FAILURE.",
    )

    # ------------------------------------------------------------------ #
    #  Validators                                                          #
    # ------------------------------------------------------------------ #

    @field_validator("skills", mode="before")
    @classmethod
    def deduplicate_skills(cls, value: list) -> list:
        """Remove duplicate skills while preserving original order."""
        seen: set = set()
        result: list = []
        for item in value:
            normalised = item.strip()
            if normalised and normalised.lower() not in seen:
                seen.add(normalised.lower())
                result.append(normalised)
        return result

    @field_validator("candidate_name", "email", "github_url", mode="before")
    @classmethod
    def empty_string_to_none(cls, value: Optional[str]) -> Optional[str]:
        """Convert empty / whitespace-only strings to ``None``."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    @property
    def filename(self) -> str:
        """Return only the filename portion of :attr:`source_file`."""
        return Path(self.source_file).name

    @property
    def is_success(self) -> bool:
        """``True`` when parsing completed without errors."""
        return self.parse_status == ParseStatus.SUCCESS

    def summary(self) -> str:
        """Return a one-line human-readable summary of the candidate."""
        name = self.candidate_name or "(unknown)"
        email = self.email or "(no email)"
        skill_count = len(self.skills)
        return (
            f"{self.filename}: {name} <{email}> | "
            f"{skill_count} skill(s) | status={self.parse_status.value}"
        )

