"""
Call Mistral API with retrieved context to generate a grounded answer.
"""
import os
# from dotenv import load_dotenv # other option
from pathlib import Path
from mistralai import Mistral

# load_dotenv() # no-op in production if .env doesn't exist; hosting platforms inject env vars directly
def get_api_key() -> str:
    """Gets mistral API key either from environment variable or from a file specified in an environment variable."""
    if key := os.environ.get("MISTRAL_API_KEY"):
        return key
    if key_file := os.environ.get("MISTRAL_API_KEY_FILE"):
        return Path(key_file).read_text().strip()
    raise RuntimeError("No Mistral API key found: set MISTRAL_API_KEY or MISTRAL_API_KEY_FILE")

MISTRAL_MODEL = "mistral-small-latest"  # adjust to preferred available model

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}, page {c.get('page')}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are an assistant answering questions about a PhD thesis.
Use ONLY the context below to answer. If the answer isn't in the context, say so.
Cite the source file and page number for claims you make.

Context:
{context_text}

Question: {query}

Answer:"""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    client = Mistral(api_key=get_api_key())
    prompt = build_prompt(query, context_chunks)

    response = client.chat.complete(
        model=MISTRAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    from retrieval import Retriever

    retriever = Retriever()
    query = input("Enter your query: ")
    chunks = retriever.search(query, top_k=5)

    answer = generate_answer(query, chunks)
    print("\n--- ANSWER ---")
    print(answer)
    print("\n--- SOURCES ---")
    for c in chunks:
        print(f"page {c.get('page')} (distance={c['distance']:.4f})")