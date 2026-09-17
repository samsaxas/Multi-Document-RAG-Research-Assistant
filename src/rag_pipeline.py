"""
End-to-end pipeline: parse documents -> chunk -> index -> retrieve -> generate.
CLI usage:
    python src/rag_pipeline.py "How many days of leave do employees get?"
    python src/rag_pipeline.py "..." --backend huggingface   # production embeddings
"""
import os
import sys
import argparse

from parse import parse_directory
from chunk import chunk_pages
from embed_store import VectorStore, get_backend

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(os.path.dirname(HERE), "data", "sample_docs")


def build_index(docs_dir: str = DOCS_DIR, backend_kind: str = "tfidf") -> VectorStore:
    pages = parse_directory(docs_dir)
    chunks = chunk_pages(pages)
    backend = get_backend(backend_kind)
    store = VectorStore(backend).build(chunks)
    print(f"[rag_pipeline] Indexed {len(chunks)} chunks from {len(pages)} pages "
          f"(backend: {backend.name})")
    return store


def ask(question: str, store: VectorStore, k: int = 4, use_llm: bool = True) -> dict:
    retrieved = store.search(question, k=k)

    if use_llm and os.environ.get("GEMINI_API_KEY"):
        from generate import generate_answer
        answer = generate_answer(question, retrieved)
    else:
        # No API key available in this environment — return the retrieved
        # context itself rather than fabricating a generated answer.
        answer = ("[No GEMINI_API_KEY set — showing retrieved context instead of a "
                   "generated answer. Set GEMINI_API_KEY to get a real Gemini response.]")

    return {"question": question, "answer": answer, "retrieved": retrieved}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--backend", default="tfidf", choices=["tfidf", "huggingface"])
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    store = build_index(backend_kind=args.backend)
    result = ask(args.question, store, k=args.k)

    print(f"\nQ: {result['question']}")
    print(f"A: {result['answer']}\n")
    print("Retrieved context:")
    for r in result["retrieved"]:
        print(f"  [{r['source']}, p.{r['page']}] score={r['score']:.3f} — {r['text'][:100]}...")
