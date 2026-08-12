"""
app.py
------
The Streamlit UI — the entry point for the entire application.

Responsibilities:
1. Let the user input a job description (paste text or upload a file).
2. Let the user upload multiple resumes.
3. Run the full pipeline: extract -> clean -> embed -> score -> rank -> explain.
4. Display results in a clean, interactive, recruiter-friendly dashboard.
"""

import streamlit as st
import pandas as pd
import spacy

from src.extractor import extract_text
from src.preprocessor import clean_text, load_skill_matcher, extract_skills
from src.embedder import ResumeEmbedder
from src.matcher import rank_candidates
from src.explainer import generate_explanation

# ---------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resource loaders.
# @st.cache_resource ensures these expensive objects (ML models) are
# loaded ONCE and reused across every user interaction/rerun, instead
# of reloading a 90MB model every time someone clicks a button.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models (first run only)...")
def load_models():
    """Loads the spaCy NLP pipeline, skill matcher, and embedding model."""
    nlp = spacy.load("en_core_web_sm")
    skill_matcher = load_skill_matcher(nlp)
    embedder = ResumeEmbedder(model_name="all-MiniLM-L6-v2")
    return nlp, skill_matcher, embedder


nlp, skill_matcher, embedder = load_models()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📄 AI-Based Resume Screening System")
st.markdown(
    "Upload a job description and a batch of resumes. The system will "
    "**rank candidates by semantic relevance** and explain exactly "
    "**why** each one matched (or didn't)."
)
st.divider()


# ---------------------------------------------------------------------------
# Input section — two columns: Job Description | Resumes
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Job Description")
    jd_input_method = st.radio(
        "Provide the job description via:",
        ["Paste text", "Upload file"],
        horizontal=True,
    )

    jd_text_raw = ""
    if jd_input_method == "Paste text":
        jd_text_raw = st.text_area(
            "Paste the job description here",
            height=250,
            placeholder="e.g. We are looking for a Python developer with "
            "experience in machine learning, REST APIs, and SQL...",
        )
    else:
        jd_file = st.file_uploader(
            "Upload job description",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
        )
        if jd_file is not None:
            jd_text_raw = extract_text(jd_file.name, jd_file.read())
            st.text_area("Extracted JD text", jd_text_raw, height=200, disabled=True)

with col2:
    st.subheader("2️⃣ Candidate Resumes")
    resume_files = st.file_uploader(
        "Upload one or more resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_uploader",
    )
    if resume_files:
        st.success(f"{len(resume_files)} resume(s) uploaded.")

st.divider()


# ---------------------------------------------------------------------------
# Run screening
# ---------------------------------------------------------------------------
run_button = st.button("🚀 Run Screening", type="primary", use_container_width=True)

if run_button:
    # --- Validation ---
    if not jd_text_raw.strip():
        st.error("Please provide a job description before running screening.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume before running screening.")
        st.stop()

    with st.spinner("Analyzing resumes... this may take a few seconds."):
        # --- Step 1: Clean the JD text and extract JD skills ---
        jd_clean = clean_text(jd_text_raw)
        jd_skills = extract_skills(jd_clean, nlp, skill_matcher)

        # --- Step 2: Extract, clean, and analyze each resume ---
        resume_names = []
        resume_texts_clean = []
        resume_skills_list = []

        for file in resume_files:
            raw_text = extract_text(file.name, file.read())
            cleaned = clean_text(raw_text)
            skills = extract_skills(cleaned, nlp, skill_matcher)

            resume_names.append(file.name)
            resume_texts_clean.append(cleaned)
            resume_skills_list.append(skills)

        # --- Step 3: Generate embeddings ---
        jd_embedding = embedder.encode([jd_clean])          # shape (1, 384)
        resume_embeddings = embedder.encode(resume_texts_clean)  # shape (n, 384)

        # --- Step 4: Rank candidates by cosine similarity ---
        ranked_results = rank_candidates(jd_embedding, resume_embeddings, resume_names)

        # --- Step 5: Build explanations for each candidate ---
        name_to_skills = dict(zip(resume_names, resume_skills_list))
        for result in ranked_results:
            explanation = generate_explanation(
                jd_skills=jd_skills,
                resume_skills=name_to_skills[result["name"]],
                similarity_score=result["score"],
            )
            result.update(explanation)

    # -----------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------
    st.success("Screening complete!")
    st.divider()
    st.subheader("📊 Ranked Candidates")

    # Summary table
    display_df = pd.DataFrame([
        {
            "Rank": i + 1,
            "Candidate": r["name"],
            "Match Score": round(r["score"] * 100, 1),
            "Skills Matched (%)": r["skill_match_percentage"],
        }
        for i, r in enumerate(ranked_results)
    ])
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bar chart visualization of match scores
    chart_df = display_df.set_index("Candidate")[["Match Score"]]
    st.bar_chart(chart_df)

    # Download results as CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        data=csv_bytes,
        file_name="resume_screening_results.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("🔍 Detailed, Explainable Insights per Candidate")

    for i, r in enumerate(ranked_results):
        with st.expander(f"#{i+1} — {r['name']}  (Score: {round(r['score']*100, 1)}%)"):
            st.write(r["summary"])

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**✅ Matched Skills**")
                st.write(", ".join(r["matched_skills"]) or "None found")
            with col_b:
                st.markdown("**❌ Missing Skills**")
                st.write(", ".join(r["missing_skills"]) or "None — full match!")
            with col_c:
                st.markdown("**➕ Extra Skills (bonus)**")
                st.write(", ".join(r["extra_skills"]) or "None")

else:
    st.info(
        "👆 Fill in the job description and upload resumes, then click "
        "**'Run Screening'** to see ranked results."
    )
