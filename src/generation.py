"""
Call Groq API with retrieved context to generate a grounded answer.
"""
import os
from groq import Groq

GROQ_MODEL = "llama-3.1-8b-instant"  # adjust to latest available free model


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return f"""You are an assistant answering questions about a PhD thesis.
Use ONLY the context below to answer. If the answer isn't in the context, say so.
Cite the source file for claims you make.

Context:
{context_text}

Question: {query}

Answer:"""


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = build_prompt(query, context_chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content