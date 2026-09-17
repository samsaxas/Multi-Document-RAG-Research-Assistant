"""
Generation layer — calls the Gemini API with retrieved chunks as context,
using a system prompt that instructs the model to answer only from the
provided context (the "grounding guardrail" from the resume bullet).

Requires: pip install google-generativeai, and a GEMINI_API_KEY env var.
This module is NOT executed in the build/test environment (no network
access to generativelanguage.googleapis.com there) — run it locally with
your own key.
"""
import os
import time
from google import genai

GROUNDING_SYSTEM_PROMPT = """You are a research assistant answering questions strictly from the provided document excerpts.

Rules:
1. Only use information explicitly present in the CONTEXT below. Do not use outside knowledge.
2. If the answer is not present in the CONTEXT, respond exactly: "I don't know based on the provided documents."
3. When you do answer, cite the source and page for every claim, like this: [source.pdf, p.3].
4. Do not speculate, infer beyond what's stated, or blend information from your own training data with the context.
"""


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = []
    for c in retrieved_chunks:
        context_blocks.append(f"[{c['source']}, p.{c['page']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"


def generate_answer(question: str, retrieved_chunks: list[dict], model_name: str | None = None) -> str:
    """Calls Gemini with the grounding guardrail. Requires GEMINI_API_KEY."""
    if model_name is None:
        model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    import google.generativeai as genai  # local import — only needed when actually calling the API

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a key from https://aistudio.google.com/apikey "
            "and set it as an environment variable before running generation."
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name, system_instruction=GROUNDING_SYSTEM_PROMPT)
    prompt = build_prompt(question, retrieved_chunks)
    import time
    for attempt in range(5):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e) or "quota" in str(e).lower():
                if attempt < 4:
                    wait_time = (attempt + 1) * 10
                    time.sleep(wait_time)
                    continue
            raise
    raise RuntimeError("Failed to generate answer after retries")
