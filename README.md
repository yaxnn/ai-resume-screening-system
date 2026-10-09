# AI Resume Screening System

A comprehensive, modular Python project that automates the screening and ranking of candidate resumes for Software Engineering (SDE) roles specializing in AI, LLMs, and Python Backend development.

The system features robust PDF/DOCX parsing, deterministic eligibility filtering, a 100-point explainable scoring engine, GitHub API enrichment, LLM-powered semantic evaluation, and an interactive Streamlit frontend.

---

## Capabilities

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
| Interactive Streamlit Frontend UI | ✅ |

---

## Project Structure

```text
resume-screening/
├── temp_resumes/            # Ephemeral directory for Streamlit uploads
├── src/
│   ├── __init__.py
│   ├── models.py            # Pydantic candidate data models
│   ├── parser.py            # PDF/DOCX text and entity extraction
│   ├── eligibility.py       # Deterministic hard-requirement filtering
│   ├── scorer.py            # Explainable 100-point scoring logic
│   ├── github_client.py     # GitHub REST API Integration
│   └── llm_evaluator.py     # Google Gen AI Semantic Evaluator
├── tests/                   # 78 passing tests ensuring complete coverage
│   ├── conftest.py        
│   ├── test_parser.py     
│   ├── test_eligibility.py
│   ├── test_scorer.py
│   ├── test_github_client.py
│   └── test_llm_evaluator.py 
├── app.py                   # Streamlit Frontend Application
├── main.py                  # CLI Application Entrypoint
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules (privacy compliant)
└── README.md
```

---

## Setup & Installation

### 1. Clone / Open the project

```bash
git clone <your-repository-url>
cd "AI Resume parsing"
```

### 2. Create a virtual environment (Recommended)

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
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

```ini
RESUME_DIR=resumes
OUTPUT_DIR=output
LOG_LEVEL=INFO

# Optional: Gemini LLM Integration
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional: GitHub API Integration
GITHUB_TOKEN=your_github_token_here
```

---

## Usage

### 🖥️ Launch the Frontend UI (Streamlit)

The recommended way to use the application is through its interactive frontend, which features file uploads, summary cards, and explainable score breakdowns.

```bash
streamlit run app.py
```
*Navigates to `http://localhost:8501` automatically.*

### 📟 CLI Basic run (Rule-based)

Place your PDF or DOCX resume files in a `resumes/` directory, then run:

```bash
python main.py
```

### 📟 CLI Run with LLM Semantic Enrichment

```bash
python main.py --use-llm
```

### 💾 Save results as JSON

```bash
python main.py --use-llm --save-json
```
This will generate timestamped JSON results containing the full candidate extraction and scoring breakdown in the `output/` directory.

---

## Testing

The project is fully unit-tested to ensure resilience and correctness. Run the tests via `pytest`:

```bash
pytest tests/ -v
```

---

## Architecture & Scoring Engine

Eligible candidates are scored out of **100 points**. By default, the system uses a highly tuned deterministic, rule-based approach. 

### Categories & Weights
1. **AI / Agentic / RAG Project Depth (40 pts)**: Penalizes tutorials or shallow API wrappers.
2. **Python & Backend Engineering (30 pts)**
3. **Cloud / Deployment / Full Stack (15 pts)**
4. **GitHub Activity (10 pts)**: Evaluates recent events and relevant public repos via GitHub API.
5. **Engineering Depth Signals (5 pts)**: Identifies testing, CI/CD, and system design patterns.

### LLM Fallback Behavior
If `--use-llm` is provided and the API key is configured, the system uses Gemini to semantically evaluate the resume for deeper insight. The LLM evaluator features automated exponential backoff retries for rate limits or network issues. If the LLM ultimately fails, or if no API key is provided, the scorer gracefully falls back to the deterministic rule-based evaluation to ensure batch processing never crashes.
