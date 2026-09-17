"""
Evaluation harness for the RAG pipeline. Two independent measurements,
matching Category 2 / Q4 of the interview prep (evaluate retrieval and
generation separately):

1. RETRIEVAL QUALITY — runs fully offline, no API key needed. For each
   labeled question, checks whether the chunk from the expected
   source/page was actually retrieved in the top-k. This produces a real
   hit-rate@k number regardless of which embedding backend is active.

2. GENERATION FAITHFULNESS / CITATION ACCURACY — requires GEMINI_API_KEY.
   For each question, calls the LLM with the grounding guardrail, then
   checks (a) whether the answer contains the expected fact
   (correctness) and (b) whether cited sources match the actual
   retrieved chunk's source (citation accuracy). This is what produces
   the "89% citation accuracy / 85% hallucination reduction" style
   numbers — and they are only real once you run this against Gemini
   with your own key. This script will not report those metrics without
   an API key; it prints a clear message instead of a fabricated number.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../src")
from parse import parse_directory
from chunk import chunk_pages
from embed_store import VectorStore, get_backend

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(os.path.dirname(HERE), "data", "sample_docs")
QA_PATH = os.path.join(HERE, "qa_set.json")


def evaluate_retrieval(store, qa_set, k=4):
    hits = 0
    evaluable = [q for q in qa_set if q["expected_source"] is not None]
    per_question = []
    for q in evaluable:
        results = store.search(q["question"], k=k)
        hit = any(
            r["source"] == q["expected_source"] and r["page"] == q["expected_page"]
            for r in results
        )
        hits += int(hit)
        per_question.append({"question": q["question"], "hit": hit,
                              "top_sources": [(r["source"], r["page"]) for r in results]})
    hit_rate = hits / len(evaluable) if evaluable else 0.0
    return {"hit_rate_at_k": round(hit_rate, 4), "k": k, "n_questions": len(evaluable),
            "per_question": per_question}


def evaluate_generation(store, qa_set, k=4):
    try:
        from generate import generate_answer
    except ImportError:
        print("[evaluate] google-generativeai not installed — skipping generation eval.")
        return None
    if not os.environ.get("GEMINI_API_KEY"):
        print("[evaluate] GEMINI_API_KEY not set — skipping generation eval. "
              "Retrieval-only metrics below are real; generation/citation metrics require your own key.")
        return None

    correct = 0
    cited_correctly = 0
    refused_correctly = 0
    results_log = []
    for q in qa_set:
        retrieved = store.search(q["question"], k=k)
        answer = generate_answer(q["question"], retrieved)
        is_refusal_case = q["expected_source"] is None
        if is_refusal_case:
            ok = "don't know" in answer.lower() or ("not" in answer.lower() and "provided" in answer.lower())
            refused_correctly += int(ok)
        else:
            contains_fact = q["expected_answer_contains"].lower() in answer.lower()
            cites_right_source = q["expected_source"] in answer
            correct += int(contains_fact)
            cited_correctly += int(cites_right_source)
        results_log.append({"question": q["question"], "answer": answer})

    n_factual = sum(1 for q in qa_set if q["expected_source"] is not None)
    n_refusal = len(qa_set) - n_factual
    return {
        "answer_correctness": round(correct / n_factual, 4) if n_factual else None,
        "citation_accuracy": round(cited_correctly / n_factual, 4) if n_factual else None,
        "correct_refusal_rate": round(refused_correctly / n_refusal, 4) if n_refusal else None,
        "log": results_log,
    }


def main(backend_kind="tfidf"):
    with open(QA_PATH) as f:
        qa_set = json.load(f)

    pages = parse_directory(DOCS_DIR)
    chunks = chunk_pages(pages)
    print(f"Indexed {len(chunks)} chunks from {len(pages)} pages using backend={backend_kind}")

    backend = get_backend(backend_kind)
    store = VectorStore(backend).build(chunks)

    retrieval_metrics = evaluate_retrieval(store, qa_set, k=4)
    print(f"\n[RETRIEVAL] hit_rate@{retrieval_metrics['k']} = "
          f"{retrieval_metrics['hit_rate_at_k']:.1%} "
          f"({retrieval_metrics['n_questions']} labeled questions)")
    for pq in retrieval_metrics["per_question"]:
        mark = "OK" if pq["hit"] else "MISS"
        print(f"  [{mark}] {pq['question'][:60]:<60} top-sources={pq['top_sources']}")

    gen_metrics = evaluate_generation(store, qa_set, k=4)
    if gen_metrics:
        print(f"\n[GENERATION] answer_correctness = {gen_metrics['answer_correctness']:.1%}")
        print(f"[GENERATION] citation_accuracy   = {gen_metrics['citation_accuracy']:.1%}")
        print(f"[GENERATION] correct_refusal_rate = {gen_metrics['correct_refusal_rate']:.1%}")

    out = {"backend": backend.name, "retrieval": retrieval_metrics, "generation": gen_metrics}
    out_path = os.path.join(os.path.dirname(HERE), "outputs", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "tfidf"
    main(kind)
