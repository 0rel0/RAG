"""
Ingestion pipeline: PDF -> chunks -> embeddings -> FAISS index.
Run this once offline to generate data/processed/index.faiss and chunks.json.
"""
import json, re
import unicodedata
import pymupdf  # orignally as import fitz but will be deprecated in future
import faiss # faiss-cpu
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PDF_DIR = Path("PhDdoc")
OUTPUT_DIR = Path("data/processed")
CHUNK_SIZE = 800       # characters, adjust after testing
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # seems to be the standard


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n[PAGE {page_num + 1}]\n" + page.get_text()
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """Normalize Unicode ligatures (e.g. 'ﬃ' -> 'ffi') and other compatibility forms."""
    return unicodedata.normalize("NFKC", text)

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Naive fixed-size chunking with overlap. TODO: improve to respect sentence/page boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text_segment = text[start:end]
        # Find the last page marker before this chunk's start
        page_matches = re.findall(r"\[PAGE (\d+)\]", text[:start + 1])
        page_number = int(page_matches[-1]) if page_matches else None
        chunks.append({"text": chunk_text_segment, "page": page_number})
        start += chunk_size - overlap
    return chunks


def build_index():
    """extracts the txt from pdf
    cuts it into chunks
    embeds the chunks using sentence-transformers
    builds index using faiss and writes it in index.faiss
    writes chunks in JSON"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    all_chunks = []  # list of dicts: {text, page, source}
    for pdf_path in PDF_DIR.glob("*.pdf"): # unecessary if there is only one source but this would scale up for more than one
        print(f"Processing {pdf_path.name}...")
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP) # all_chunks = chunks for now (1 source)
        for chunk in chunks:
            all_chunks.append({**chunk, "source": pdf_path.name}) # Merge the chunk dict with the source, instead of nesting it

    print(f"Total chunks: {len(all_chunks)}")

    print("Embedding chunks...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
    with open(OUTPUT_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Saved index and metadata to {OUTPUT_DIR}/")


if __name__ == "__main__":
    build_index()