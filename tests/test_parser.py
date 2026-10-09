"""
test_parser.py — Unit tests for src.parser and src.models (Phase 1).

Test coverage:
  - Successful PDF parsing (name, email, GitHub, skills, projects)
  - Successful DOCX parsing
  - Empty / blank PDF → parse failure with correct status
  - Malformed PDF → parse failure with correct status
  - Unsupported file extension → parse failure with correct status
  - Candidate model field validation (deduplication, empty→None)
  - parse_all_resumes: batch behaviour, failure isolation, deduplication
  - discover_resume_files: missing directory error
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import Candidate, ParseStatus
from src.parser import (
    _extract_candidate_name,
    _extract_email,
    _extract_github_url,
    _extract_projects,
    _extract_skills,
    discover_resume_files,
    parse_all_resumes,
    parse_resume,
)


# ===========================================================================
# Helpers / constants
# ===========================================================================

FULL_TEXT = """\
Alice Johnson
alice.johnson@example.com
https://github.com/alicejohnson

Skills:
Python, Machine Learning, SQL, Docker, Git

Projects:
Resume Screening Tool
Sentiment Analysis API
"""


# ===========================================================================
# Unit tests: extraction helpers
# ===========================================================================


class TestExtractEmail:
    def test_valid_email(self) -> None:
        assert _extract_email("Contact: alice@example.com") == "alice@example.com"

    def test_no_email(self) -> None:
        assert _extract_email("No contact info here.") is None

    def test_email_with_subdomain(self) -> None:
        assert _extract_email("me@mail.uni.edu") == "me@mail.uni.edu"


class TestExtractGithub:
    def test_https_url(self) -> None:
        assert _extract_github_url("https://github.com/alicejohnson") == "https://github.com/alicejohnson"

    def test_url_with_trailing_path(self) -> None:
        # Only the profile root should be returned
        assert _extract_github_url("https://github.com/alicejohnson/my-project") == "https://github.com/alicejohnson"

    def test_no_github(self) -> None:
        assert _extract_github_url("https://gitlab.com/alice") is None


class TestExtractName:
    def test_proper_name_first_line(self) -> None:
        text = "Alice Johnson\nalice@example.com\n"
        assert _extract_candidate_name(text) == "Alice Johnson"

    def test_no_name(self) -> None:
        text = "alice@example.com\nhttps://github.com/alice\n"
        assert _extract_candidate_name(text) is None

    def test_three_word_name(self) -> None:
        text = "John Michael Doe\njohn@example.com"
        assert _extract_candidate_name(text) == "John Michael Doe"


class TestExtractSkills:
    def test_comma_separated(self) -> None:
        text = "Skills:\nPython, SQL, Docker\n\nOther"
        skills = _extract_skills(text)
        assert "Python" in skills
        assert "SQL" in skills

    def test_no_skills_section(self) -> None:
        text = "Alice Johnson\nalice@example.com"
        assert _extract_skills(text) == []


class TestExtractProjects:
    def test_projects_extracted(self) -> None:
        text = "Projects:\nResume Screening Tool\nSentiment Analysis API\n\nExperience"
        projects = _extract_projects(text)
        assert "Resume Screening Tool" in projects
        assert "Sentiment Analysis API" in projects

    def test_no_projects_section(self) -> None:
        text = "Alice Johnson\nalice@example.com"
        assert _extract_projects(text) == []


# ===========================================================================
# Unit tests: Candidate model
# ===========================================================================


class TestCandidateModel:
    def test_required_field(self) -> None:
        c = Candidate(source_file="/some/file.pdf")
        assert c.source_file == "/some/file.pdf"
        assert c.parse_status == ParseStatus.SUCCESS
        assert c.skills == []
        assert c.projects == []

    def test_skills_deduplicated(self) -> None:
        c = Candidate(
            source_file="/f.pdf",
            skills=["Python", "python", "SQL", "Python"],
        )
        assert c.skills.count("Python") == 1
        assert len(c.skills) == 2  # Python + SQL

    def test_empty_string_fields_become_none(self) -> None:
        c = Candidate(
            source_file="/f.pdf",
            candidate_name="  ",
            email="",
            github_url="   ",
        )
        assert c.candidate_name is None
        assert c.email is None
        assert c.github_url is None

    def test_failure_status(self) -> None:
        c = Candidate(
            source_file="/f.pdf",
            parse_status=ParseStatus.FAILURE,
            parse_error="Something went wrong",
        )
        assert not c.is_success
        assert c.parse_error == "Something went wrong"

    def test_filename_property(self) -> None:
        c = Candidate(source_file="/path/to/resume.pdf")
        assert c.filename == "resume.pdf"

    def test_summary_contains_name_and_status(self) -> None:
        c = Candidate(
            source_file="/path/to/resume.pdf",
            candidate_name="Alice Johnson",
            email="alice@example.com",
            skills=["Python", "SQL"],
        )
        summary = c.summary()
        assert "Alice Johnson" in summary
        assert "success" in summary
        assert "2 skill" in summary


# ===========================================================================
# Integration tests: parse_resume (single file)
# ===========================================================================


class TestParseResumePDF:
    def test_successful_pdf_parse(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert candidate.parse_status == ParseStatus.SUCCESS
        assert candidate.is_success
        assert len(candidate.extracted_text) > 0

    def test_pdf_email_extracted(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert candidate.email == "alice.johnson@example.com"

    def test_pdf_github_extracted(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert candidate.github_url == "https://github.com/alicejohnson"

    def test_pdf_name_extracted(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert candidate.candidate_name == "Alice Johnson"

    def test_pdf_skills_extracted(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert len(candidate.skills) > 0

    def test_pdf_projects_extracted(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert len(candidate.projects) > 0

    def test_source_file_recorded(self, sample_pdf: Path) -> None:
        candidate = parse_resume(sample_pdf)
        assert candidate.source_file == str(sample_pdf.resolve())


class TestParseResumeMinimalPDF:
    def test_minimal_pdf_succeeds(self, minimal_pdf: Path) -> None:
        candidate = parse_resume(minimal_pdf)
        assert candidate.parse_status == ParseStatus.SUCCESS

    def test_minimal_pdf_no_github(self, minimal_pdf: Path) -> None:
        candidate = parse_resume(minimal_pdf)
        assert candidate.github_url is None

    def test_minimal_pdf_empty_skills(self, minimal_pdf: Path) -> None:
        candidate = parse_resume(minimal_pdf)
        assert candidate.skills == []


class TestParseResumeDOCX:
    def test_successful_docx_parse(self, sample_docx: Path) -> None:
        candidate = parse_resume(sample_docx)
        assert candidate.parse_status == ParseStatus.SUCCESS
        assert len(candidate.extracted_text) > 0

    def test_docx_email_extracted(self, sample_docx: Path) -> None:
        candidate = parse_resume(sample_docx)
        assert candidate.email == "alice.johnson@example.com"


class TestParseResumeEmptyPDF:
    def test_empty_pdf_fails(self, empty_pdf: Path) -> None:
        candidate = parse_resume(empty_pdf)
        assert candidate.parse_status == ParseStatus.FAILURE
        assert not candidate.is_success
        assert candidate.parse_error is not None

    def test_empty_pdf_text_is_empty(self, empty_pdf: Path) -> None:
        candidate = parse_resume(empty_pdf)
        assert candidate.extracted_text == ""


class TestParseResumeMalformed:
    def test_malformed_pdf_fails(self, malformed_pdf: Path) -> None:
        candidate = parse_resume(malformed_pdf)
        assert candidate.parse_status == ParseStatus.FAILURE
        assert candidate.parse_error is not None


class TestParseResumeUnsupported:
    def test_unsupported_extension_fails(self, unsupported_file: Path) -> None:
        candidate = parse_resume(unsupported_file)
        assert candidate.parse_status == ParseStatus.FAILURE
        assert "Unsupported" in (candidate.parse_error or "")


# ===========================================================================
# Integration tests: parse_all_resumes (batch)
# ===========================================================================


class TestParseAllResumes:
    def test_returns_candidates_and_failures(self, resume_directory: Path) -> None:
        candidates, failures = parse_all_resumes(resume_directory)
        assert isinstance(candidates, list)
        assert isinstance(failures, dict)

    def test_correct_total_count(self, resume_directory: Path) -> None:
        """The resume_directory fixture contains 4 files."""
        candidates, _ = parse_all_resumes(resume_directory)
        assert len(candidates) == 4

    def test_at_least_two_successes(self, resume_directory: Path) -> None:
        """sample_pdf, minimal_pdf, and sample_docx should all succeed."""
        candidates, _ = parse_all_resumes(resume_directory)
        successes = [c for c in candidates if c.is_success]
        assert len(successes) >= 2

    def test_empty_pdf_is_in_failures(self, resume_directory: Path) -> None:
        _, failures = parse_all_resumes(resume_directory)
        assert "empty_resume.pdf" in failures

    def test_one_bad_file_does_not_stop_batch(self, resume_directory: Path) -> None:
        """All files must be processed even if one fails."""
        candidates, _ = parse_all_resumes(resume_directory)
        # We expect all 4 files to have a Candidate record
        assert len(candidates) == 4

    def test_no_duplicate_candidates(self, resume_directory: Path) -> None:
        candidates, _ = parse_all_resumes(resume_directory)
        paths = [c.source_file for c in candidates]
        assert len(paths) == len(set(paths))


class TestDiscoverResumeFiles:
    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            list(discover_resume_files(missing))

    def test_only_supported_extensions_returned(self, resume_directory: Path) -> None:
        # plant a .txt file in the directory; it must be ignored
        txt_file = resume_directory / "ignore_me.txt"
        txt_file.write_text("ignore", encoding="utf-8")
        files = list(discover_resume_files(resume_directory))
        names = [f.name for f in files]
        assert "ignore_me.txt" not in names

