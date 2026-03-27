"""FAISS-based vector memory store for per-session context."""
from __future__ import annotations
import os
import json
import logging
import pickle
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

DISTANCE_THRESHOLD = 1.5  # L2 distance; lower = more similar


class FAISSMemoryStore:
    """
    Per-session FAISS IndexFlatL2 memory.
    Embeddings are generated with sentence-transformers (all-MiniLM-L6-v2).
    Indexes are persisted to disk under persist_dir/{session_id}/.
    """

    def __init__(self, persist_dir: str = "./memory") -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._indexes: dict[str, object] = {}   # session_id -> faiss.Index
        self._texts: dict[str, list[str]] = {}  # session_id -> list of stored texts
        self._embedder = None  # lazy-loaded

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    def store(self, session_id: str, text: str, metadata: dict | None = None) -> None:
        """Embed text and add it to the session's FAISS index."""
        if not text.strip():
            return

        index, texts = self._get_or_create(session_id)
        embedding = self._embed([text])  # shape: (1, 384)

        import faiss
        index.add(embedding)
        texts.append(text)
        self._persist(session_id)

    def retrieve(self, session_id: str, query: str, top_k: int = 5) -> list[str]:
        """Return the top_k most semantically similar stored texts for the query."""
        if session_id not in self._indexes or not self._texts.get(session_id):
            return []

        index = self._indexes[session_id]
        texts = self._texts[session_id]
        query_embedding = self._embed([query])

        k = min(top_k, len(texts))
        distances, indices = index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and dist < DISTANCE_THRESHOLD:
                results.append(texts[idx])
        return results

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        return embeddings.astype("float32")

    def _get_or_create(self, session_id: str):
        if session_id not in self._indexes:
            # Try loading from disk first
            loaded = self._load(session_id)
            if loaded:
                self._indexes[session_id], self._texts[session_id] = loaded
            else:
                import faiss
                dim = 384  # all-MiniLM-L6-v2 output dimension
                self._indexes[session_id] = faiss.IndexFlatL2(dim)
                self._texts[session_id] = []
        return self._indexes[session_id], self._texts[session_id]

    def _persist(self, session_id: str) -> None:
        import faiss
        session_dir = self.persist_dir / session_id
        session_dir.mkdir(exist_ok=True)
        faiss.write_index(self._indexes[session_id], str(session_dir / "index.faiss"))
        with open(session_dir / "texts.pkl", "wb") as f:
            pickle.dump(self._texts[session_id], f)

    def _load(self, session_id: str) -> tuple | None:
        import faiss
        session_dir = self.persist_dir / session_id
        index_path = session_dir / "index.faiss"
        texts_path = session_dir / "texts.pkl"
        if not index_path.exists() or not texts_path.exists():
            return None
        try:
            index = faiss.read_index(str(index_path))
            with open(texts_path, "rb") as f:
                texts = pickle.load(f)
            return index, texts
        except Exception as exc:
            logger.warning("Failed to load FAISS index for session %s: %s", session_id, exc)
            return None
