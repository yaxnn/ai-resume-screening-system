import streamlit as st
import json
import os
import glob
import subprocess
from pathlib import Path

# Important to add src path for parsing
import sys
sys.path.append(os.path.dirname(__file__))
from src.parser import parse_resume

st.set_page_config(page_title="AI Resume Screener", layout="wide", initial_sidebar_state="collapsed")

# Simple, professional styling based on requirements (Light bg, purple accent)
st.markdown("""
<style>
    :root {
        --primary-color: #8a2be2;
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #7a1fd1;
        color: white;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI Resume Screener")

st.write("Upload candidate resumes (PDF or DOCX) to screen them against the Python + AI/LLM criteria.")
uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Run Screening"):
    if not uploaded_files:
        st.warning("Please upload at least one resume to run the screening.")
    else:
        with st.spinner("Running screening pipeline (this may take a minute)..."):
            # Save uploaded files to a temporary directory
            temp_dir = Path("temp_resumes")
            temp_dir.mkdir(exist_ok=True)
            for f in uploaded_files:
                file_path = temp_dir / f.name
                file_path.write_bytes(f.getbuffer())
                
            subprocess.run([sys.executable, "main.py", "--resume-dir", str(temp_dir), "--save-json"], capture_output=True, text=True)
            st.success("Screening complete!")

output_dir = "output"
latest_file = None
if os.path.exists(output_dir):
    files = glob.glob(os.path.join(output_dir, "scoring_results_*.json"))
    if files:
        latest_file = max(files, key=os.path.getmtime)

if not latest_file:
    st.info("No screening results found. Click 'Run Screening' to start.")
else:
    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        valid_candidates = [c for c in data if c.get("parse_status") == "success"]
        total = len(valid_candidates)
        eligible = [c for c in valid_candidates if c["eligibility"]["eligible"]]
        rejected = [c for c in valid_candidates if not c["eligibility"]["eligible"]]

        # Summary Cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Resumes parsed", total)
        col2.metric("Eligible Candidates", len(eligible))
        col3.metric("Rejected Candidates", len(rejected))

        st.markdown("---")

        # View Toggle
        view = st.radio("View Candidates:", ["Eligible", "Rejected", "All"], horizontal=True)

        if view == "Eligible":
            display_data = eligible
        elif view == "Rejected":
            display_data = rejected
        else:
            display_data = valid_candidates

        if display_data:
            display_data.sort(key=lambda x: x["scoring"].get("total_score", 0), reverse=True)

            table_rows = []
            for c in display_data:
                is_elig = c["eligibility"]["eligible"]
                rank = c["scoring"].get("rank", "-") if is_elig else "Rejected"
                table_rows.append({
                    "Rank / Status": rank,
                    "Name": c["candidate_name"] or "(unknown)",
                    "Score": c["scoring"].get("total_score", 0) if is_elig else "-",
                    "GitHub": c["github_url"] or "-"
                })

            st.dataframe(table_rows, use_container_width=True)

            st.markdown("### Candidate Details")
            candidate_names = [f"{c['candidate_name'] or '(unknown)'} - {c['filename']}" for c in display_data]
            selected_name = st.selectbox("Select a candidate:", candidate_names)

            if selected_name:
                idx = candidate_names.index(selected_name)
                selected_c = display_data[idx]
                elig = selected_c.get("eligibility", {})
                scoring = selected_c.get("scoring", {})
                is_eligible = elig.get("eligible", False)

                col_info, col_score = st.columns(2)
                
                # We need to grab the source skills and projects if available
                cand_skills = []
                cand_projects = []
                source_path = Path(selected_c["source_file"])
                if source_path.exists():
                    cand_obj = parse_resume(source_path)
                    cand_skills = cand_obj.skills
                    cand_projects = cand_obj.projects

                with col_info:
                    st.write(f"**Email:** {selected_c['email']}")
                    st.write(f"**GitHub:** {selected_c['github_url']}")
                    
                    if is_eligible:
                        st.success("Status: **ELIGIBLE**")
                    else:
                        st.error("Status: **REJECTED**")
                        st.markdown("#### Rejection Reasons")
                        for r in elig.get("rejection_reasons", []):
                            st.write(f"- ❌ {r}")
                        
                        st.markdown("#### Requirement Check")
                        python_ev = elig.get("python_evidence", [])
                        ai_ev = elig.get("ai_evidence", [])
                        
                        if python_ev:
                            st.write("✅ Demonstrated Python usage")
                        else:
                            st.write("❌ Missing Python evidence")
                            
                        if ai_ev:
                            st.write("✅ Demonstrated AI/LLM/RAG/Agentic evidence")
                        else:
                            st.write("❌ Missing AI/LLM/RAG/Agentic evidence")
                            
                    st.markdown("#### Matching Skills")
                    matched_skills = elig.get("matched_skills", [])
                    if matched_skills:
                        st.markdown(" ".join([f'<span style="background-color: var(--primary-color); color: white; padding: 0.2rem 0.5rem; border-radius: 0.5rem; margin-right: 0.5rem; display: inline-block;">{s}</span>' for s in matched_skills]), unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.write("These skills match the target requirements. Evidence:")
                        for ev in elig.get("python_evidence", []) + elig.get("ai_evidence", []):
                            st.write(f"- {ev}")
                    else:
                        st.write("No matching target skills found.")
                        
                    if not is_eligible and cand_skills:
                        # List other skills that did match if any (this is what matched_skills shows)
                        other_skills = [s for s in cand_skills if s not in matched_skills]
                        if other_skills:
                            with st.expander("Other extracted skills"):
                                st.write(", ".join(other_skills))
                                
                    st.markdown("#### Project Evidence")
                    if cand_projects:
                        for p in cand_projects:
                            st.write(f"- {p}")
                    else:
                        st.write("*(No clear project descriptions found in resume)*")
                        
                with col_score:
                    if is_eligible:
                        st.markdown(f"### Total Score: {scoring.get('total_score', 0)} / 100")
                        st.write(f"**Rank:** {scoring.get('rank', '-')} out of {len([c for c in display_data if c.get('eligibility', {}).get('eligible')])}")
                        
                        bd = scoring.get("breakdown", {})
                        ce = scoring.get("category_explanations", {})
                        
                        st.markdown("#### Score Breakdown & Explanations")
                        
                        categories = [
                            ("AI / Agentic / RAG Project Depth", "ai_project_depth", 40),
                            ("Python & Backend Engineering", "python_backend", 30),
                            ("Cloud / Deployment / Full Stack", "cloud_fullstack", 15),
                            ("GitHub Activity", "github", 10),
                            ("Engineering Depth Signals", "engineering_depth", 5)
                        ]
                        
                        for label, key, max_pts in categories:
                            pts = bd.get(key, 0)
                            with st.expander(f"{label}: {pts} / {max_pts} pts", expanded=(pts > 0)):
                                if key == "github":
                                    status = scoring.get("github_enrichment_status", "unavailable")
                                    if status == "success":
                                        st.write("✅ GitHub data successfully retrieved.")
                                        if scoring.get("github_activity_summary"):
                                            st.write(f"**Activity:** {scoring['github_activity_summary']}")
                                        if scoring.get("github_repo_summary"):
                                            st.write(f"**Repositories:** {scoring['github_repo_summary']}")
                                    else:
                                        st.warning(f"GitHub data {status}. (This does not affect other scores)")
                                
                                expl = ce.get(key, {})
                                strengths = expl.get("strengths", [])
                                concerns = expl.get("concerns", [])
                                evidence = expl.get("evidence", [])
                                
                                if strengths:
                                    st.write("**Points Awarded For:**")
                                    for s in strengths:
                                        st.write(f"- ✅ {s}")
                                        
                                if concerns:
                                    st.write("**Points Lost / Weaknesses:**")
                                    for c in concerns:
                                        st.write(f"- ⚠️ {c}")
                                        
                                if evidence:
                                    st.write("**Supporting Evidence:**")
                                    for e in evidence:
                                        st.write(f"- 📄 {e}")
                                        
                                if not strengths and not concerns and not evidence:
                                    st.write("*(No specific evidence found for this category)*")

                    else:
                        st.write("### Score: N/A")
                        st.info("Candidate did not meet hard eligibility requirements and is therefore not scored or ranked.")

        st.markdown("---")
        with open(latest_file, "r", encoding="utf-8") as f:
            st.download_button(
                label="Download Results (JSON)",
                data=f,
                file_name=os.path.basename(latest_file),
                mime="application/json"
            )
            
    except Exception as e:
        st.error(f"Error loading results: {e}")

