"""
explainer.py
------------
Responsible for ONE job: turning raw similarity scores into
human-readable, defensible explanations.

This is the difference between a "black box" AI project and an
industry-grade one. Recruiters (and your mentors) need to know
*why* a candidate ranked where they did — not just a number.
"""


def generate_explanation(jd_skills: set, resume_skills: set, similarity_score: float) -> dict:
    """
    Compares the skill sets of the JD and a resume, and builds a
    structured explanation.

    Args:
        jd_skills: Set of skills extracted from the job description.
        resume_skills: Set of skills extracted from this resume.
        similarity_score: Cosine similarity score (0 to 1) for this resume.

    Returns:
        A dictionary containing:
        - matched_skills: skills present in both JD and resume
        - missing_skills: skills required by JD but absent in resume
        - extra_skills: skills in resume not mentioned in JD (bonus skills)
        - skill_match_percentage: % of JD skills that were matched
        - summary: a plain-English one-line summary
    """
    matched_skills = sorted(jd_skills.intersection(resume_skills))
    missing_skills = sorted(jd_skills.difference(resume_skills))
    extra_skills = sorted(resume_skills.difference(jd_skills))

    if jd_skills:
        skill_match_percentage = round(
            (len(matched_skills) / len(jd_skills)) * 100, 1
        )
    else:
        # If the JD has no recognizable skills, fall back purely on
        # the semantic similarity score.
        skill_match_percentage = 0.0

    # Build a human-readable summary sentence
    if similarity_score >= 0.75:
        strength = "a strong overall match"
    elif similarity_score >= 0.55:
        strength = "a moderate match"
    else:
        strength = "a weak match"

    summary = (
        f"This candidate is {strength} for the role "
        f"(semantic similarity: {similarity_score:.2f}), matching "
        f"{len(matched_skills)} of {len(jd_skills)} required skills "
        f"({skill_match_percentage}%)."
    )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "skill_match_percentage": skill_match_percentage,
        "summary": summary,
    }
