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
CHUNK_OVERLAP_SENTENCES = 2
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # seems to be the standard

FLOATING_ACCENT_FIXES = {
    "´e": "é", "´E": "É",
    "`e": "è", "`E": "È",
    "^e": "ê", "^E": "Ê", "ˆe": "ê",
    " ̧c": "ç", " ̧C": "Ç",
    "`a": "à", "ˆa": "â",
    "ˆo": "ô",
}

def extract_pages_from_pdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract raw text from a PDF file, per page.
    Returns a list of (page_number, page_text) tuples, 1-indexed."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        page_text = normalize_text(page.get_text())
        pages.append((page_num + 1, page_text))
    return pages


def normalize_text(text: str) -> str:
    """ Fixes the "floating" french accent issues from the old PDF
    Normalize Unicode ligatures (e.g. 'ﬃ' -> 'ffi') and other compatibility forms."""
    for broken, fixed in FLOATING_ACCENT_FIXES.items():
        new_text = text.replace(broken, fixed)
        text=new_text
    return unicodedata.normalize("NFKC", text)

def chunk_text(pages: list[tuple[int, str]], max_chunk_size: int, overlap_sentences: int = 1) -> list[dict]:
    """Sentence-aware chunking across pages, with cross-page overlap.
    Groups sentences until max_chunk_size is reached, keeping chunks variable
    in length but never splitting mid-sentence. Tracks the page of the first
    sentence in each chunk."""

    # Flatten all pages into a single list of (sentence, page_number) tuples
    sentence_page_pairs = []
    for page_num, page_text in pages:
        sentences = re.split(r'(?<=[.!?])\s+', page_text.strip()) #split by sentence
        for s in sentences:
            if s.strip():
                sentence_page_pairs.append((s.strip(), page_num))

    chunks = []
    current_list = []  # list of (sentence, page_num)
    current_length = 0

    def flush_chunk():
        """utility function to create the chunk made of joined sentences with overlap
        flow processing, so extra checks for first time etc."""
        if not current_list:
            return
        chunk_str = " ".join(s for s, _ in current_list).strip()
        if chunk_str:
            first_page = current_list[0][1]
            chunks.append({"text": chunk_str, "page": first_page})

    for sentence, page_num in sentence_page_pairs:
        sentence_len = len(sentence)
        if current_length + sentence_len > max_chunk_size and current_list:
            flush_chunk()
            current_list = current_list[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) for s, _ in current_list)

        current_list.append((sentence, page_num))
        current_length += sentence_len

    flush_chunk() #last go once out of the loop
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
        text = extract_pages_from_pdf(pdf_path)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP_SENTENCES) # all_chunks = chunks for now (1 source)
        for chunk in chunks:
            if chunk["text"].strip(): # extra check in case of emtpy chunks after splitting by sentence
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