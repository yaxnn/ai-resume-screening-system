"""
conftest.py — Shared pytest fixtures for Phase 1 tests.

Fixtures create minimal but realistic test files (PDF and DOCX) on-the-fly
using reportlab and python-docx so the test suite has no dependency on
external resume files from the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_RESUME_TEXT = """\
Alice Johnson
alice.johnson@example.com
https://github.com/alicejohnson

Skills:
Python, Machine Learning, SQL, Docker, Git

Projects:
Resume Screening Tool
Sentiment Analysis API

Experience:
Software Engineer at TechCorp (2021–2024)
"""

MINIMAL_RESUME_TEXT = """\
Bob Smith
bob.smith@example.org
"""


def _make_pdf(path: Path, text: str) -> None:
    """Write a single-page PDF containing *text* to *path* using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas

    c = rl_canvas.Canvas(str(path), pagesize=A4)
    text_object = c.beginText(50, 800)
    text_object.setFont("Helvetica", 10)
    for line in text.splitlines():
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()


def _make_docx(path: Path, text: str) -> None:
    """Write a DOCX file containing *text* (one paragraph per line) to *path*."""
    from docx import Document

    doc = Document()
    for line in text.splitlines():
        doc.add_paragraph(line)
    doc.save(str(path))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped temporary directory for all test fixture files."""
    return tmp_path_factory.mktemp("fixtures")


@pytest.fixture(scope="session")
def sample_pdf(fixtures_dir: Path) -> Path:
    """A valid single-page PDF resume with all common fields."""
    path = fixtures_dir / "sample_resume.pdf"
    _make_pdf(path, SAMPLE_RESUME_TEXT)
    return path


@pytest.fixture(scope="session")
def minimal_pdf(fixtures_dir: Path) -> Path:
    """A valid PDF resume with only name and email (no skills/projects/GitHub)."""
    path = fixtures_dir / "minimal_resume.pdf"
    _make_pdf(path, MINIMAL_RESUME_TEXT)
    return path


@pytest.fixture(scope="session")
def sample_docx(fixtures_dir: Path) -> Path:
    """A valid DOCX resume with all common fields."""
    path = fixtures_dir / "sample_resume.docx"
    _make_docx(path, SAMPLE_RESUME_TEXT)
    return path


@pytest.fixture(scope="session")
def empty_pdf(fixtures_dir: Path) -> Path:
    """A PDF file that contains no text (blank page only)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas

    path = fixtures_dir / "empty_resume.pdf"
    c = rl_canvas.Canvas(str(path), pagesize=A4)
    c.showPage()
    c.save()
    return path


@pytest.fixture(scope="session")
def malformed_pdf(fixtures_dir: Path) -> Path:
    """A file with a .pdf extension but corrupt/invalid binary content."""
    path = fixtures_dir / "malformed_resume.pdf"
    path.write_bytes(b"%PDF-1.4\nthis is not a real pdf\x00\xff\xfe")
    return path


@pytest.fixture(scope="session")
def unsupported_file(fixtures_dir: Path) -> Path:
    """A plain text file with an unsupported extension."""
    path = fixtures_dir / "resume.txt"
    path.write_text("John Doe\njohn@example.com\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def resume_directory(
    fixtures_dir: Path,
    sample_pdf: Path,
    minimal_pdf: Path,
    sample_docx: Path,
    empty_pdf: Path,
) -> Path:
    """A temporary directory pre-populated with fixture resume files.

    Contains: one full PDF, one minimal PDF, one DOCX, one blank PDF.
    The empty PDF is expected to fail parsing.
    """
    resume_dir = fixtures_dir / "resumes"
    resume_dir.mkdir(exist_ok=True)
    import shutil

    for src in (sample_pdf, minimal_pdf, sample_docx, empty_pdf):
        shutil.copy(src, resume_dir / src.name)
    return resume_dir

