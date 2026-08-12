"""
embedder.py
-----------
Responsible for ONE job: turning text into embeddings using a
pretrained Sentence-Transformer model.

Why Sentence-Transformers instead of raw BERT?
Raw BERT produces one embedding PER TOKEN, and naively averaging them
gives poor sentence-level similarity. Sentence-Transformers are
fine-tuned specifically so that the FULL-TEXT embedding is directly
comparable via cosine similarity — which is exactly what we need here.

We use 'all-MiniLM-L6-v2': a small (80MB), fast model that still
achieves strong semantic similarity performance — ideal for a
resource-constrained demo/production environment.
"""

from sentence_transformers import SentenceTransformer
import numpy as np


class ResumeEmbedder:
    """
    A thin wrapper around a SentenceTransformer model.
    Wrapping it in a class (instead of using the model directly)
    means we can swap the underlying model later without changing
    any calling code — good software engineering practice.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Loads the pretrained model. This is the expensive step
        (downloads + loads weights into memory), so we only want
        to do it ONCE per app session — handled via caching in app.py.

        Args:
            model_name: Name of the Hugging Face sentence-transformers model.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Converts a list of texts into a matrix of embeddings.

        Args:
            texts: List of strings (e.g., resume texts).

        Returns:
            A NumPy array of shape (num_texts, embedding_dim).
            For 'all-MiniLM-L6-v2', embedding_dim = 384.
        """
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # Pre-normalize so cosine sim = dot product
        )
        return embeddings
