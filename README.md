# AI Resume Screening System

A modular Python project that reads candidate resumes and extracts structured
information from them. This is **Phase 1** of a multi-phase AI-powered resume
screening and ranking pipeline.

---

## Phase 5 Capabilities

| Feature | Status |
|---|---|
| PDF & DOCX text extraction | ✅ |
| Conservative entity extraction (name, email, github) | ✅ |
| Deterministic Eligibility Filtering (Python + AI) | ✅ |
| Explainable 100-point Scoring Engine | ✅ |
| AI / RAG Project Depth Scoring (Max 40) | ✅ |
| Python & Backend Engineering Scoring (Max 30) | ✅ |
| Cloud / Deployment / Full Stack Scoring (Max 15) | ✅ |
| Engineering Depth Signals Scoring (Max 5) | ✅ |
| GitHub API Enrichment & Activity Scoring (Max 10) | ✅ |
| LLM Semantic Enrichment (`google-genai`) | ✅ |
| LLM-based Scoring Overrides & Project Summaries | ✅ |
| Automatic Fallback to Rule-based Scoring on LLM Failure | ✅ |
| Candidate Ranking (Eligible only) | ✅ |
| Pydantic data validation | ✅ |
| Configurable directories & Optional JSON export | ✅ |

> **Not included in Phase 5:** Frontend UI, advanced web scraping.

---

## Project Structure

```
resume-screening/
├── resumes/               
├── src/
│   ├── __init__.py
│   ├── models.py          
│   ├── parser.py          
│   ├── eligibility.py     
│   ├── scorer.py          
│   ├── github_client.py   ← GitHub REST API Integration
│   └── llm_evaluator.py   ← Google Gen AI Semantic Evaluator
├── tests/
│   ├── conftest.py        
│   ├── test_parser.py     
│   ├── test_eligibility.py
│   ├── test_scorer.py     
│   └── test_llm_evaluator.py ← Unit tests for LLM integration
├── output/                
├── main.py                
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone / open the project

```bash
cd "AI Resume parsing"
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env` and adjust as needed:

```bash
copy .env.example .env
```

```ini
RESUME_DIR=resumes
OUTPUT_DIR=output
LOG_LEVEL=INFO

# Phase 4: Gemini LLM Integration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

### 5. Add resume files

Place your PDF or DOCX resume files in the `resumes/` directory.

---

## Usage

### Basic run (Rule-based, prints to terminal)

```bash
python main.py
```

### Run with LLM Semantic Enrichment

```bash
python main.py --use-llm
```

### Save results as JSON

```bash
python main.py --use-llm --save-json
```
This will save two files in the output directory:
- `eligibility_results_<timestamp>.json`
- `scoring_results_<timestamp>.json` (Contains LLM evaluations, summaries, and citations)

---

## Running Tests

```bash
pytest tests/ -v
```

### Launch the Frontend UI (Streamlit)

```bash
streamlit run app.py
```

---

## Phase 4: LLM-Enriched Scoring Engine

Eligible candidates are scored out of **100 points**. By default, the system uses a deterministic, rule-based approach. If `--use-llm` is provided and the API key is configured, the system uses Gemini to semantically evaluate the resume.

### Categories & Weights
1. **AI / Agentic / RAG Project Depth (40 pts)**: LLM evaluates project depth, heavily penalizing tutorials or API wrappers.
2. **Python & Backend Engineering (30 pts)**
3. **Cloud / Deployment / Full Stack (15 pts)**
4. **GitHub Activity (10 pts)**: Evaluates recent events and relevant repos via GitHub API.
5. **Engineering Depth Signals (5 pts)**

### LLM Fallback Behavior
The LLM evaluator features automated exponential backoff retries for rate limits or network issues. If the LLM ultimately fails, or if no API key is provided, the scorer gracefully falls back to the deterministic rule-based evaluation to ensure batch processing never crashes.

---

## Dependencies

| Package | Purpose |
|---|---|
| `PyMuPDF` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `pydantic` | Structured data validation |
| `python-dotenv` | `.env` configuration loading |
| `google-genai` | Gemini LLM integration |
| `pytest`, `pytest-cov`, `reportlab` | Testing & Coverage |

---

## Roadmap

- **Phase 1** — File Parsing & Structured Extraction
- **Phase 2** — Deterministic Eligibility Filtering
- **Phase 3** — Rule-based Explainable Scoring & Ranking
- **Phase 4** — LLM-powered enrichment & semantic evaluation
- **Phase 5** — GitHub API integration & Frontend
