# RAG-Based Multi-Document Research Analyst

A retrieval-augmented generation pipeline that answers questions across
multiple PDF documents, grounded strictly in retrieved context, with a
built-in evaluation harness for retrieval quality and citation accuracy.

## Pipeline

```
PDFs -> parse.py (per-page text extraction, source/page metadata kept)
     -> chunk.py (overlapping chunks, ~700 chars, 100 char overlap)
     -> embed_store.py (embeddings -> FAISS index)
     -> rag_pipeline.py (top-k retrieval -> Gemini generation)
     -> generate.py (grounding guardrail: "answer only from context,
                      say I don't know otherwise, cite source+page")
```

## Two embedding backends — read this before citing numbers

This repo supports two backends in `src/embed_store.py`:

- **`huggingface`** (production — `sentence-transformers/all-MiniLM-L6-v2`):
  this is what should be cited as "HuggingFace embeddings." Requires
  internet access to the HuggingFace Hub to download the model on first run.
- **`tfidf`** (local fallback): scikit-learn TF-IDF + SVD, no downloads
  required. This exists **only** to verify the parsing/chunking/indexing/
  retrieval logic works end-to-end in network-restricted environments. It
  is explicitly labeled in code and output as "NOT the production
  HuggingFace backend" — do not cite results from this backend as
  HuggingFace-based.

Run with `--backend huggingface` (or set it in the Streamlit sidebar) once
you have normal internet access, to use the actual production backend.

## Evaluation (`eval/evaluate.py`)

Two metrics, matching how RAG systems should actually be evaluated
(retrieval and generation are separate failure modes — see interview
prep, Category 2 Q4):

1. **Retrieval hit-rate@k** — runs fully offline. For each of 16 labeled
   questions (`eval/qa_set.json`), checks whether the chunk from the
   correct source/page was retrieved in the top-4. Verified working in
   this repo: **100% hit-rate@4** on the 3-document sample corpus
   (`outputs/eval_results.json`).
2. **Citation accuracy / answer correctness / correct-refusal-rate** —
   requires `GEMINI_API_KEY`. Calls Gemini with the grounding guardrail
   for each question, checks whether the answer contains the expected
   fact and cites the correct source, and separately checks whether the
   one deliberately unanswerable question in the set is correctly
   refused ("I don't know based on the provided documents") rather than
   hallucinated. **This is where the citation-accuracy and
   hallucination-reduction numbers come from — run this yourself with
   your own Gemini key to get real figures for your resume, rather than
   reusing a number from someone else's run.**

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"   # from https://aistudio.google.com/apikey

# Generate the 3 sample PDFs (or drop your own PDFs into data/sample_docs/)
python data/make_sample_docs.py

# Ask a single question via CLI
python src/rag_pipeline.py "How many days of leave do employees get?" --backend huggingface

# Full evaluation (retrieval always runs; generation runs if GEMINI_API_KEY is set)
python eval/evaluate.py huggingface

# Chat UI
streamlit run app.py
```

## Repo structure

```
rag_research_analyst/
├── data/
│   ├── make_sample_docs.py   # generates 3 sample PDFs with known facts
│   └── sample_docs/          # (or drop your own PDFs here)
├── src/
│   ├── parse.py              # PDF -> per-page text + metadata
│   ├── chunk.py               # overlapping chunking
│   ├── embed_store.py        # HuggingFace / TF-IDF embeddings + FAISS
│   ├── generate.py           # Gemini call + grounding guardrail prompt
│   └── rag_pipeline.py       # ties retrieval + generation together (CLI)
├── eval/
│   ├── qa_set.json           # 16 labeled Q&A pairs (incl. 1 unanswerable)
│   └── evaluate.py           # retrieval + generation evaluation harness
├── outputs/
│   └── eval_results.json     # actual run output (retrieval metrics)
├── app.py                    # Streamlit chat UI
└── requirements.txt
```

## Test Questions

Use the following questions to manually test retrieval, multi-document reasoning,
citations, and refusal behavior.

### 1. Direct factual questions

1. How many days of leave do employees get?
2. What is the company's work-from-home policy?
3. What are the standard working hours?
4. What is the probation period?
5. What benefits are provided to employees?

### 2. Source-specific questions

6. What does the HR policy document say about leave?
7. According to the employee handbook, what are the working hours?
8. Which document describes the company's remote-work policy?
9. On which page is the leave policy mentioned?

### 3. Cross-document questions

10. What are the differences between the leave policies mentioned in the documents?
11. Which document provides information about employee benefits?
12. Compare the working-hours policies across the available documents.

### 4. Paraphrased questions

13. If an employee wants to take time off, how much leave are they entitled to?
14. Can employees work remotely according to the provided documents?
15. What is the expected daily working schedule?

### 5. Unanswerable / hallucination test

16. What is the company's revenue for the previous financial year?

Expected behavior:
The system should respond that the information cannot be found
in the provided documents rather than inventing an answer.