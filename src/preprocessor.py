"""
preprocessor.py
----------------
Responsible for TWO jobs:
1. Cleaning raw text lightly (for embedding — we keep it close to
   natural language, since transformer models are trained on real
   sentences, not stripped-down keyword soup).
2. Extracting a structured set of "skills" from text using a
   predefined skill dictionary + spaCy's PhraseMatcher. This powers
   the explainability layer later.
"""

import re
import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# A curated list of common technical + soft skills.
# In a real production system, this list would live in a database or be
# learned from a labeled dataset — here we hardcode a solid starting set
# so the app works out of the box.
# ---------------------------------------------------------------------------
SKILL_KEYWORDS = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r",
    "go", "rust", "php", "scala", "kotlin", "swift",
    # AI / ML
    "machine learning", "deep learning", "natural language processing",
    "nlp", "computer vision", "pytorch", "tensorflow", "keras",
    "scikit-learn", "transformers", "huggingface", "opencv", "llm",
    "generative ai", "data science", "neural networks",
    # Web / backend
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "rest api", "graphql", "html", "css", "spring boot",
    # Data / cloud
    "pandas", "numpy", "spark", "hadoop", "aws", "azure", "gcp",
    "docker", "kubernetes", "airflow", "etl", "data pipeline",
    "power bi", "tableau", "excel",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    # Practices / tools
    "git", "github", "ci/cd", "agile", "scrum", "jira", "linux",
    "unit testing", "microservices",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "project management", "time management", "critical thinking",
    "collaboration", "presentation", "stakeholder management",
]


def clean_text(text: str) -> str:
    """
    Lightly cleans text before embedding generation.

    We intentionally do NOT aggressively strip stopwords or lemmatize
    here, because sentence-transformer models expect natural, fluent
    text to produce the best embeddings. Over-cleaning would actually
    HURT embedding quality.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text with normalized whitespace and no weird symbols.
    """
    # Remove email addresses and URLs (they add noise, not meaning)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove excessive special characters but keep basic punctuation
    text = re.sub(r"[^a-zA-Z0-9.,;:()/#+\-\s]", " ", text)
    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_skill_matcher(nlp: spacy.language.Language) -> PhraseMatcher:
    """
    Builds a spaCy PhraseMatcher pre-loaded with our skill vocabulary.
    This lets us find multi-word skills (e.g., "machine learning") in
    text efficiently, in a single pass.

    Args:
        nlp: A loaded spaCy language pipeline.

    Returns:
        A configured PhraseMatcher instance.
    """
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in SKILL_KEYWORDS]
    matcher.add("SKILLS", patterns)
    return matcher


def extract_skills(text: str, nlp: spacy.language.Language, matcher: PhraseMatcher) -> set:
    """
    Scans text and returns the set of known skills found in it.

    Args:
        text: Text to scan (resume or job description).
        nlp: Loaded spaCy pipeline.
        matcher: PhraseMatcher built by load_skill_matcher().

    Returns:
        A set of matched skill strings (lowercase, deduplicated).
    """
    doc = nlp(text.lower())
    matches = matcher(doc)
    found_skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        found_skills.add(span.text.strip())
    return found_skills
