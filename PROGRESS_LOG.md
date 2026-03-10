# Project Progress Log

## Pipeline 1 — Cells 1-9 — ✅ Complete
PDF processing, chunking, embeddings, ChromaDB storage

## Pipeline 2 — Cells 10-14 — ✅ Complete
Retrieval, prompt building, Llama-3.1 integration

---

## Key Fixes Made
- Groq model: use `llama-3.1-8b-instant` (not `llama-3-8b-8192` which is decommissioned)
- LangChain import: use `langchain_text_splitters` (not `langchain.text_splitter`)
- Cell 9 updated: now smart save/load checkpoint
- GitHub fix: run JSON cleanup cell before pushing to fix widget metadata

---

## Baseline Performance (Pipeline 2)
- Average latency: 0.58 seconds
- Confidence: High on 4/5 questions
- Chunks per query: 4
- Embedding dimensions: 384

---

## Next Up
- Phase 3: Hybrid Search + Reranker + Conditional Query Expansion
