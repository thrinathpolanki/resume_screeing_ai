"""
matcher.py
----------
Responsible for ONE job: comparing embeddings and producing a ranked
list of candidates.

We use cosine similarity because it measures the ANGLE between two
vectors, not their magnitude. This matters because resume length
varies wildly (a 1-page resume vs. a 3-page resume) — cosine
similarity focuses purely on semantic direction/meaning, not on
how much text was written.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def rank_candidates(
    jd_embedding: np.ndarray,
    resume_embeddings: np.ndarray,
    resume_names: list[str],
) -> list[dict]:
    """
    Scores each resume against the job description and returns them
    sorted from best match to worst match.

    Args:
        jd_embedding: Embedding vector for the job description, shape (1, dim).
        resume_embeddings: Embedding matrix for all resumes, shape (n, dim).
        resume_names: Filenames/identifiers corresponding to each resume row.

    Returns:
        List of dicts sorted by descending score:
        [{"name": str, "score": float}, ...]
    """
    # cosine_similarity expects 2D arrays; jd_embedding is already (1, dim)
    similarities = cosine_similarity(jd_embedding, resume_embeddings)[0]

    results = [
        {"name": name, "score": float(score)}
        for name, score in zip(resume_names, similarities)
    ]

    # Sort descending by score (best match first)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
