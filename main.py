"""
main.py — Command-line entry point for the AI Resume Screening System.

Phase 1: Discovers, parses, and reports on resumes found in a configurable
input directory.  Results are printed to stdout and optionally saved as a
JSON file in the output directory.

Usage
-----
    python main.py [--resume-dir PATH] [--output-dir PATH] [--save-json]

Environment variables (loaded from .env if present)
----------------------------------------------------
    RESUME_DIR   Path to the directory containing resume files (default: resumes)
    OUTPUT_DIR   Path to the directory for output files        (default: output)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load .env before importing project modules that read env vars
# ---------------------------------------------------------------------------
load_dotenv()

# Ensure stdout/stderr use UTF-8 on Windows (avoids UnicodeEncodeError with
# box-drawing characters and emoji when the terminal code page is cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.models import ParseStatus  # noqa: E402
from src.parser import parse_all_resumes  # noqa: E402

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Resume Screening System — Phase 1: Parse & Extract",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--resume-dir",
        default=os.getenv("RESUME_DIR", "resumes"),
        help="Directory containing resume files (PDF / DOCX).",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "output"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=False,
        help="Save parsed results to a JSON file in --output-dir.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=False,
        help="Use LLM for Phase 4 semantic evaluation of candidates.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run Phase 1-4: discover, parse, filter, and score all resumes."""
    args = build_arg_parser().parse_args(argv)

    resume_dir = Path(args.resume_dir)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------ #
    #  Banner                                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  AI Resume Screening System — Phase 1-4")
    print("=" * 60)
    print(f"  Resume directory : {resume_dir.resolve()}")
    print(f"  Output directory : {output_dir.resolve()}")
    print(f"  Use LLM scoring  : {args.use_llm}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------ #
    #  Parse & Eligibility                                                #
    # ------------------------------------------------------------------ #
    candidates, failures = parse_all_resumes(resume_dir)

    if not candidates:
        print("No resume files found or all files failed discovery.")
        return 1
        
    from src.eligibility import check_all_eligibility
    from src.eligibility import check_eligibility
    from src.scorer import score_and_rank_candidates

    # We need EligibilityResult objects for scoring, so we will generate them directly here
    # or reconstruct. Wait, check_all_eligibility returns a list of Dicts.
    # Let's map check_eligibility to get the actual objects.
    candidates_with_elig = []
    for cand in candidates:
        elig_result = check_eligibility(cand)
        candidates_with_elig.append((cand, elig_result))
    
    # ------------------------------------------------------------------ #
    #  Scoring & Ranking (Phase 3 & 4)                                    #
    # ------------------------------------------------------------------ #
    scored_results = score_and_rank_candidates(candidates_with_elig, use_llm=args.use_llm)

    # ------------------------------------------------------------------ #
    #  Report                                                             #
    # ------------------------------------------------------------------ #
    successes = [c for c in candidates if c.parse_status == ParseStatus.SUCCESS]
    failed = [c for c in candidates if c.parse_status == ParseStatus.FAILURE]
    
    eligible_count = sum(1 for c, e, s in scored_results if e.eligible)
    rejected_count = len(successes) - eligible_count

    print(f"{'─'*60}")
    print(f"  Total discovered : {len(candidates)}")
    print(f"  Parsed OK        : {len(successes)}")
    print(f"  Failed           : {len(failed)}")
    print(f"  Eligible         : {eligible_count}")
    print(f"  Rejected         : {rejected_count}")
    print(f"{'─'*60}\n")

    # Successful candidates
    if successes:
        print("✅  Successfully parsed candidates (Eligibility & Ranking):\n")
        # Sort output by rank if ranked, then by name
        def sort_key(item):
            cand, elig, scoring = item
            rank = scoring.rank if scoring.rank is not None else 999999
            name = cand.candidate_name or ""
            return (rank, name)

        for cand, elig, scoring in sorted(scored_results, key=sort_key):
            if cand.parse_status != ParseStatus.SUCCESS:
                continue
            name = cand.candidate_name or "(unknown)"
            email = cand.email or "(no email)"
            if elig.eligible:
                elig_status = f"ELIGIBLE | Rank: {scoring.rank} | Score: {scoring.total_score}/100"
                print(f"  {cand.filename}: {name} <{email}> | {elig_status}")
            else:
                elig_status = "REJECTED"
                print(f"  {cand.filename}: {name} <{email}> | {elig_status}")
                reasons = " | ".join(elig.rejection_reasons)
                print(f"    └─ Rejected: {reasons}")
        print()

    # Failures
    if failures:
        print("❌  Failed files:\n")
        for filename, reason in failures.items():
            print(f"  {filename}: {reason}")
        print()

    # ------------------------------------------------------------------ #
    #  Optional JSON export                                               #
    # ------------------------------------------------------------------ #
    if args.save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Phase 2 export structure (backward compatibility)
        elig_export = []
        # Phase 3 export structure
        score_export = []

        for cand, elig, scoring in scored_results:
            base_info = {
                "source_file": cand.source_file,
                "filename": cand.filename,
                "candidate_name": cand.candidate_name,
                "email": cand.email,
                "github_url": cand.github_url,
                "parse_status": cand.parse_status.value,
            }
            
            elig_dict = {**base_info, "eligibility": elig.to_dict()}
            elig_export.append(elig_dict)

            score_dict = {**base_info, "eligibility": elig.to_dict(), "scoring": scoring.to_dict()}
            score_export.append(score_dict)

        # Save Phase 2
        elig_path = output_dir / f"eligibility_results_{timestamp}.json"
        with open(elig_path, "w", encoding="utf-8") as fh:
            json.dump(elig_export, fh, indent=2, ensure_ascii=False)
        print(f"💾  Phase 2 Eligibility results saved to: {elig_path}")

        # Save Phase 3
        score_path = output_dir / f"scoring_results_{timestamp}.json"
        with open(score_path, "w", encoding="utf-8") as fh:
            json.dump(score_export, fh, indent=2, ensure_ascii=False)
        print(f"💾  Phase 3 Scoring results saved to: {score_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
