"""NanoVDR-based visual document retriever.

Loads pre-indexed document embeddings and uses NanoVDR-S-Multi
for query encoding. Returns top-k relevant document pages.
"""

import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DocumentRetriever:
    def __init__(
        self,
        model_name: str = "nanovdr/NanoVDR-S-Multi",
        index_dir: str = "data/index",
    ):
        print(f"Loading retriever: {model_name}")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        print(f"  Model loaded in {time.time() - t0:.1f}s")

        # Load pre-computed document embeddings
        index_path = Path(index_dir)
        self.doc_embeddings = np.load(index_path / "doc_embeddings.npy").astype(np.float32)
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.doc_embeddings = self.doc_embeddings / norms

        # Load page metadata (page_id → image path mapping)
        import json
        with open(index_path / "page_metadata.json") as f:
            self.page_metadata = json.load(f)

        print(f"  Index: {len(self.page_metadata)} pages, dim={self.doc_embeddings.shape[1]}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k document pages for a text query.

        Returns list of dicts with keys: page_id, score, image_path, metadata
        """
        t0 = time.time()
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = (query_emb @ self.doc_embeddings.T)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        latency = time.time() - t0

        results = []
        for idx in top_indices:
            page = self.page_metadata[idx]
            results.append({
                "page_id": page.get("page_id", f"page_{idx}"),
                "score": float(scores[idx]),
                "image_path": page.get("image_path", ""),
                "source": page.get("source", ""),
                "page_number": page.get("page_number", idx),
            })

        return results, latency
