"""
Load FAISS index + chunk metadata, and perform similarity search for a query.
"""
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
INDEX_FILE = DATA_DIR / "index.faiss"
CHUNKS_FILE = DATA_DIR / "chunks.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class Retriever:
    """class to load FAISS index and perform similarity search for a query."""
    def __init__(self):
        """takes no arguments, the file paths are stored as globals
        reads index, loads chunks anf initializes the embedding model"""
        self.index = faiss.read_index(str(INDEX_FILE))
        with open(CHUNKS_FILE) as f:
            self.chunks = json.load(f)
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """takes a query string and returns the top_k most similar chunks from the index"""
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "distance": float(dist)})
        return results

# for manual testing
if __name__ == "__main__":
    retriever = Retriever()
    query = input("Enter your query: ")
    results = retriever.search(query, top_k=5)
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (distance={r['distance']:.4f}, page={r.get('page')}) ---")
        print(r["text"][:300])