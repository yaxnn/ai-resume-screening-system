"""
parser.py — Resume file discovery and text-extraction logic .

Responsibilities
----------------
* Discover all supported resume files in a configurable directory.
* Extract plain text from PDF (via PyMuPDF) and DOCX (via python-docx) files.
* Populate a :class:`~src.models.Candidate` record for each file.
* Extract basic candidate fields conservatively using simple regex patterns.
* Handle errors at the per-file level so that one bad file never aborts the
  entire batch.
* Avoid duplicate processing of the same file (by resolved absolute path).

This module contains **no** scoring, LLM calls, GitHub API calls, or CLI
logic — those belong to other modules or to ``main.py``.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Tuple

from src.models import Candidate, ParseStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".pdf", ".docx")

# ---------------------------------------------------------------------------
# Conservative extraction patterns
# ---------------------------------------------------------------------------

# E-mail pattern — standard RFC-5321 local part, common TLDs
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)

# GitHub profile URL — https://github.com/<username>  (no sub-paths captured)
_GITHUB_RE = re.compile(
    r"https?://(?:www\.)?github\.com/([A-Za-z0-9\-]+)(?:/[^\s]*)?",
    re.IGNORECASE,
)

# Skills section header — looks for a heading that says "Skills" (or similar)
_SKILLS_SECTION_RE = re.compile(
    r"(?:technical\s+)?skills?[\s:–\-]*\n(.*?)(?=\n{2,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Projects section header
_PROJECTS_SECTION_RE = re.compile(
    r"projects?[\s:–\-]*\n(.*?)(?=\n{2,}|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Common skill separators inside a skills block
_SKILL_SPLIT_RE = re.compile(r"[,|•·;\n]+")

# Name heuristic: first non-empty, non-email, short line that looks like a name
# (2–4 capitalised words, optionally with hyphens/apostrophes)
_NAME_LINE_RE = re.compile(
    r"^([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+){1,3})\s*$",
)

# ALL-CAPS name pattern: "SHIVAM RAJ", "M SAI KUSHITH" (2–5 words, ≤40 chars)
_NAME_ALLCAPS_RE = re.compile(
    r"^([A-Z][A-Z'\-]*(?:\s+[A-Z][A-Z'\-]*){1,4})\s*$",
)

# Common false-positive phrases that look like names but aren't.
# Matched case-insensitively against candidate lines.
_NAME_BLOCKLIST: Set[str] = {
    # Job titles / roles
    "software developer", "software engineer", "software engineer intern",
    "full stack developer", "frontend developer", "backend developer",
    "web developer", "data scientist", "data analyst", "data engineer",
    "machine learning engineer", "ml engineer", "devops engineer",
    "associate software engineer", "senior software engineer",
    "junior software engineer", "gen ai developer intern",
    "gen ai developer", "ai developer", "ai engineer",
    "python developer", "java developer", "react developer",
    # Section headers
    "professional summary", "summary", "objective", "profile summary",
    "skills summary", "career objective", "profile objective",
    "technical skills", "work experience", "education", "projects",
    "certifications", "achievements", "experience", "contact",
    "programming language", "programming languages",
    "react native basic",
    # Institutions / companies (common false positives from the dataset)
    "webtrex software services", "dirghayu bharat services",
    "vijayawada nalanda junior college",
}



# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _extract_text_pdf(path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF (fitz).

    Parameters
    ----------
    path:
        Absolute path to the PDF file.

    Returns
    -------
    str
        All page text joined with newlines.

    Raises
    ------
    ImportError
        When PyMuPDF is not installed.
    Exception
        On any file-level read error.
    """
    try:
        import pymupdf as fitz  # PyMuPDF (modern import; `fitz` is deprecated)
    except ModuleNotFoundError:
        try:
            import fitz  # fallback for older PyMuPDF installs
        except ModuleNotFoundError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF parsing.  "
                "Install it with: pip install PyMuPDF"
            ) from exc

    doc = fitz.open(str(path))
    pages: List[str] = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _extract_text_docx(path: Path) -> str:
    """Extract text from a DOCX file using python-docx.

    Parameters
    ----------
    path:
        Absolute path to the DOCX file.

    Returns
    -------
    str
        All paragraph text joined with newlines.

    Raises
    ------
    ImportError
        When python-docx is not installed.
    Exception
        On any file-level read error.
    """
    try:
        from docx import Document  # python-docx
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX parsing.  "
            "Install it with: pip install python-docx"
        ) from exc

    doc = Document(str(path))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def extract_text(path: Path) -> str:
    """Dispatch to the appropriate extractor based on file extension.

    Parameters
    ----------
    path:
        Absolute path to the resume file.

    Returns
    -------
    str
        Extracted plain text.  May be empty for blank or image-only files.

    Raises
    ------
    ValueError
        When the file extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_text_pdf(path)
    if suffix == ".docx":
        return _extract_text_docx(path)
    raise ValueError(f"Unsupported file type: '{suffix}' (file: {path.name})")


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------


def _extract_email(text: str) -> Optional[str]:
    """Return the first e-mail address found in *text*, or ``None``."""
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def _extract_github_url(text: str) -> Optional[str]:
    """Return the first GitHub profile URL found in *text*, or ``None``."""
    match = _GITHUB_RE.search(text)
    if match:
        # Reconstruct a clean profile URL (strip any trailing path)
        username = match.group(1)
        return f"https://github.com/{username}"
    return None


def _extract_candidate_name(text: str) -> Optional[str]:
    """Attempt to extract a candidate name from the first lines of *text*.

    Uses a conservative heuristic with two passes:
        1. Look for an ALL-CAPS name line (common in Indian resume formats).
        2. Look for a title-case name line (e.g. "Alice Johnson").

    Lines matching common job titles, section headers, or known institution
    names are excluded via a blocklist.  Returns ``None`` if no match is
    found — never invents a value.

    Parameters
    ----------
    text:
        Full extracted text of the resume.
    """
    candidates_lines: List[str] = []
    for line in text.splitlines()[:25]:  # inspect only the first 25 lines
        line = line.strip()
        if not line or len(line) > 60:
            continue
        # Skip lines that look like e-mails or URLs
        if _EMAIL_RE.search(line) or _GITHUB_RE.search(line):
            continue
        # Skip lines containing phone numbers (digits + dashes/spaces)
        if re.search(r"\+?\d[\d\s\-]{7,}", line):
            continue
        # Skip lines with pipe separators (contact info rows)
        if line.count("|") >= 2:
            continue
        # Skip blocklisted phrases
        if line.lower().strip() in _NAME_BLOCKLIST:
            continue
        candidates_lines.append(line)

    # Pass 1: ALL-CAPS names (e.g. "SHIVAM RAJ", "PAVANI A")
    for line in candidates_lines:
        # Must be short and ALL-CAPS (allowing single-letter initials)
        if line == line.upper() and len(line.split()) <= 5:
            match = _NAME_ALLCAPS_RE.match(line)
            if match:
                # Title-case the result for display consistency
                name = match.group(1).strip()
                return name.title()

    # Pass 2: Title-case names (e.g. "Alice Johnson")
    for line in candidates_lines:
        if line.isupper():
            continue  # already handled above
        match = _NAME_LINE_RE.match(line)
        if match:
            return match.group(1)
    return None


def _extract_skills(text: str) -> List[str]:
    """Extract skills from a skills section if one is present.

    Uses an improved section-boundary heuristic: the skills block ends at
    the next recognisable section heading (e.g. 'Experience:', 'Projects:',
    'Education:') or at a double-newline, whichever comes first.

    Returns an empty list when no recognisable skills section is found.
    """
    # Match "Skills", "Technical Skills", etc. and capture until the next
    # section heading or double-newline.
    skills_section_re = re.compile(
        r"(?:technical\s+)?skills?\s*(?:[:–\-]\s*|\n)"  # header
        r"(.*?)"  # body (non-greedy)
        r"(?="  # lookahead for end boundary
        r"\n\s*(?:experience|projects?|education|work\s+experience|"
        r"professional\s+experience|employment|certifications?|"
        r"achievements?|publications?|summary|objective|"
        r"profile|hobbies|interests|languages?(?:\s+proficiency)?|"
        r"references?|awards?|courses?|training|activities)"
        r"\s*(?:[:–\-]|\n)"
        r"|\n{2,}|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    match = skills_section_re.search(text)
    if not match:
        return []
    block = match.group(1)
    raw = _SKILL_SPLIT_RE.split(block)
    skills: List[str] = []
    for item in raw:
        item = item.strip().strip("–-•·*:()[]")
        # Filter: skip very short items, long sentence-like strings,
        # and sub-headings (ending with ':')
        if not item or len(item) <= 1 or len(item) > 40:
            continue
        if item.endswith(":"):
            continue
        skills.append(item)
        if len(skills) >= 30:  # reasonable cap
            break
    return skills


def _extract_projects(text: str) -> List[str]:
    """Extract project entries from a projects section if one is present.

    Returns an empty list when no recognisable projects section is found.
    Each non-blank line in the projects block is treated as a separate entry.
    """
    match = _PROJECTS_SECTION_RE.search(text)
    if not match:
        return []
    block = match.group(1)
    projects: List[str] = []
    for line in block.splitlines():
        line = line.strip().strip("–-•·*")
        if line and len(line) > 3:
            projects.append(line)
            if len(projects) >= 10:  # cap at 10 entries
                break
    return projects


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_resume_files(directory: Path) -> Generator[Path, None, None]:
    """Yield all supported resume files found directly in *directory*.

    Only the top-level directory is scanned (non-recursive) so that the
    ``resumes/`` folder stays flat.  Subdirectories are silently skipped.

    Parameters
    ----------
    directory:
        Path to the directory to scan.

    Yields
    ------
    Path
        Absolute path to each supported file.

    Raises
    ------
    FileNotFoundError
        When *directory* does not exist.
    NotADirectoryError
        When *directory* is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Resume directory not found: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory}")

    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _file_fingerprint(path: Path) -> str:
    """Return a stable identity key for *path* (its resolved absolute path)."""
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------


def parse_resume(path: Path) -> Candidate:
    """Parse a single resume file and return a populated :class:`Candidate`.

    All field extraction is best-effort and conservative.  Missing fields are
    left as ``None`` / empty list — never filled with invented values.

    Parameters
    ----------
    path:
        Absolute (or relative) path to the resume file.

    Returns
    -------
    Candidate
        Populated record.  On failure, ``parse_status`` is set to
        :attr:`~src.models.ParseStatus.FAILURE` and ``parse_error`` contains
        a human-readable description.
    """
    source_str = str(path.resolve())

    # ------------------------------------------------------------------ #
    #  Unsupported extension guard                                        #
    # ------------------------------------------------------------------ #
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return Candidate(
            source_file=source_str,
            parse_status=ParseStatus.FAILURE,
            parse_error=(
                f"Unsupported file type: '{path.suffix}'.  "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # ------------------------------------------------------------------ #
    #  Text extraction                                                    #
    # ------------------------------------------------------------------ #
    try:
        text = extract_text(path)
    except Exception as exc:
        logger.warning("Failed to extract text from '%s': %s", path.name, exc)
        return Candidate(
            source_file=source_str,
            parse_status=ParseStatus.FAILURE,
            parse_error=f"Text extraction failed: {exc}",
        )

    if not text.strip():
        logger.warning("'%s' produced empty text (blank or image-only file).", path.name)
        return Candidate(
            source_file=source_str,
            extracted_text="",
            parse_status=ParseStatus.FAILURE,
            parse_error="File produced no extractable text (may be blank or image-only).",
        )

    # ------------------------------------------------------------------ #
    #  Field extraction                                                   #
    # ------------------------------------------------------------------ #
    return Candidate(
        source_file=source_str,
        extracted_text=text,
        candidate_name=_extract_candidate_name(text),
        email=_extract_email(text),
        github_url=_extract_github_url(text),
        skills=_extract_skills(text),
        projects=_extract_projects(text),
        parse_status=ParseStatus.SUCCESS,
    )


def parse_all_resumes(
    directory: Path,
) -> Tuple[List[Candidate], Dict[str, str]]:
    """Parse every supported resume in *directory* and return results.

    Files are deduplicated by resolved absolute path so that symlinks or
    duplicate entries do not produce duplicate :class:`Candidate` records.

    Parameters
    ----------
    directory:
        Path to the directory containing resume files.

    Returns
    -------
    candidates : list[Candidate]
        One record per file (including failures).
    failures : dict[str, str]
        Mapping of ``filename → error_message`` for every failed parse.
    """
    candidates: List[Candidate] = []
    failures: Dict[str, str] = {}
    seen: Set[str] = set()

    try:
        files = list(discover_resume_files(directory))
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error("Cannot scan resume directory: %s", exc)
        return [], {}

    logger.info("Discovered %d file(s) in '%s'.", len(files), directory)

    for path in files:
        fingerprint = _file_fingerprint(path)
        if fingerprint in seen:
            logger.debug("Skipping duplicate file: %s", path.name)
            continue
        seen.add(fingerprint)

        logger.info("Parsing: %s", path.name)
        candidate = parse_resume(path)
        candidates.append(candidate)

        if not candidate.is_success:
            failures[path.name] = candidate.parse_error or "Unknown error"
            logger.warning("Parse failed for '%s': %s", path.name, candidate.parse_error)

    return candidates, failures
