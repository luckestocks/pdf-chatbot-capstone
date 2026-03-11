"""
PDF Chatbot — Phase 7 Streamlit UI
Capstone Project by luckestocks
Built on top of Phases 1–6 pipeline
"""

import streamlit as st
import os
import time
import hashlib
import pickle
import numpy as np

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── API KEY — Replace this string with your actual Groq key ─────────────────
# ⚠️  TO USE st.secrets INSTEAD (recommended for production):
#     1. In Streamlit Cloud → App Settings → Secrets, add:
#           GROQ_API_KEY = "gsk_your_key_here"
#     2. Change the line below to:
#           GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_KEY = "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"   # <── PASTE YOUR KEY HERE
# ─────────────────────────────────────────────────────────────────────────────

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ─── Lazy imports (cached so they only install once per session) ──────────────
@st.cache_resource(show_spinner="Loading models and libraries…")
def load_dependencies():
    import PyPDF2
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    from rank_bm25 import BM25Okapi
    from groq import Groq
    return PyPDF2, SentenceTransformer, CrossEncoder, chromadb, BM25Okapi, Groq

PyPDF2, SentenceTransformer, CrossEncoder, chromadb, BM25Okapi, Groq = load_dependencies()

# ─── Load embedding + reranker models (cached) ───────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en")

@st.cache_resource(show_spinner="Loading reranker model…")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

embed_model = load_embed_model()
reranker    = load_reranker()
groq_client = Groq(api_key=GROQ_API_KEY)

# ─── Pipeline Functions ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove garbled unicode, math symbols, and noise from PDF text."""
    import re
    # Replace common garbled characters and math symbols
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    # Remove non-printable / control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Remove isolated unicode math/symbol characters (not regular letters/numbers)
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_text_from_pdf(uploaded_file) -> tuple[str, int]:
    """Extract all text from an uploaded PDF. Returns (text, page_count)."""
    reader = PyPDF2.PdfReader(uploaded_file)
    pages  = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(clean_text(text))
    return "\n\n".join(pages), len(reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Generate embeddings for all chunks."""
    return embed_model.encode(chunks, show_progress_bar=False)


def build_chromadb(chunks: list[str], embeddings: np.ndarray):
    """Create an in-memory ChromaDB collection from chunks + embeddings."""
    client     = chromadb.Client()
    collection = client.get_or_create_collection("pdf_chunks")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(chunks))],
    )
    return collection


def build_bm25(chunks: list[str]):
    """Build a BM25 index from chunks."""
    tokenised = [c.lower().split() for c in chunks]
    return BM25Okapi(tokenised)


def retrieve_chunks(query: str, collection, embed_model, top_k: int = 6) -> list[dict]:
    """Semantic retrieval from ChromaDB."""
    q_emb    = embed_model.encode([query]).tolist()
    results  = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs     = results["documents"][0]
    return [{"text": d, "source": "semantic"} for d in docs]


def bm25_retrieve(query: str, bm25, chunks: list[str], top_k: int = 6) -> list[dict]:
    """BM25 keyword retrieval."""
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"text": chunks[i], "bm25_score": float(scores[i]), "source": "bm25"} for i in top_idx]


def hybrid_search(query: str, collection, bm25, chunks: list[str],
                  embed_model, top_k: int = 6) -> list[dict]:
    """Merge semantic + BM25 results (deduped by text)."""
    semantic = retrieve_chunks(query, collection, embed_model, top_k)
    keyword  = bm25_retrieve(query, bm25, chunks, top_k)
    seen, merged = set(), []
    for item in semantic + keyword:
        key = item["text"][:80]
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def full_hybrid_retrieve(query: str, collection, bm25, chunks: list[str],
                         embed_model, reranker, top_k: int = 4) -> list[dict]:
    """Hybrid search → CrossEncoder reranking → top_k results."""
    candidates = hybrid_search(query, collection, bm25, chunks, embed_model, top_k=6)
    pairs      = [[query, c["text"]] for c in candidates]
    scores     = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the LLM prompt from retrieved chunks."""
    context_parts = []
    for i, c in enumerate(chunks):
        score = c.get("rerank_score", c.get("confidence", 0.0))
        context_parts.append(f"[Chunk {i+1} | score={score:.3f}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_parts)
    return f"""Use the context below to answer the question.
If the context contains a clear answer, use it.
If the context is garbled or contains only math with no readable answer, use general knowledge for basic definitions only.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""


def _is_garbled(chunks: list[dict]) -> bool:
    """Return True if the top chunk is mostly garbled/math content."""
    if not chunks:
        return True
    top_text = chunks[0].get("text", "")
    # Count non-ASCII characters (math symbols, etc.)
    non_ascii = sum(1 for c in top_text if ord(c) > 127)
    ratio = non_ascii / max(len(top_text), 1)
    # Also check if top rerank score is very low (poor retrieval)
    top_score = chunks[0].get("rerank_score", 1.0)
    return ratio > 0.05 or top_score < 0.01


def get_answer(query: str, collection, bm25, chunks: list[str],
               embed_model, reranker, answer_cache: dict) -> dict:
    """Full pipeline: cache check → hybrid retrieve → rerank → LLM answer."""
    cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()
    if cache_key in answer_cache:
        cached = answer_cache[cache_key]
        cached["from_cache"] = True
        return cached

    t0        = time.time()
    retrieved = full_hybrid_retrieve(query, collection, bm25, chunks, embed_model, reranker)

    # ── Garbled chunk detection: skip document context if chunks are unusable ─
    if _is_garbled(retrieved):
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": (
                    "You are a helpful assistant. Answer clearly and concisely "
                    "in 2-4 sentences. Give a direct, accurate answer."
                 )},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        answer = response.choices[0].message.content.strip()
        answer += "\n\n*(Note: answered from general knowledge — the relevant document section contained mathematical notation that could not be parsed.)*"
    else:
        # ── Normal path: answer from document context ─────────────────────────
        prompt = build_prompt(query, retrieved)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system",
                 "content": (
                    "You are a strict document QA assistant. "
                    "Answer only from the provided context. "
                    "If the answer is not present, say so clearly. "
                    "Never invent or assume facts. "
                    "Keep your answer concise — maximum 4 sentences."
                 )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()

    latency = round(time.time() - t0, 2)
    result = {
        "answer":     answer,
        "chunks":     retrieved,
        "latency":    latency,
        "from_cache": False,
    }
    answer_cache[cache_key] = result
    return result


# ─── Session State Init ───────────────────────────────────────────────────────
for key, default in {
    "messages":     [],
    "answer_cache": {},
    "pdf_loaded":   False,
    "chunks":       [],
    "collection":   None,
    "bm25":         None,
    "pdf_name":     "",
    "page_count":   0,
    "chunk_count":  0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chatbot")
    st.caption("Capstone Project — luckestocks")
    st.divider()

    uploaded = st.file_uploader(
        "Upload a PDF to begin",
        type=["pdf"],
        help="Upload any PDF. The app will extract text, build embeddings, and prepare the chatbot.",
    )

    if uploaded and (not st.session_state.pdf_loaded or uploaded.name != st.session_state.pdf_name):
        with st.spinner("📖 Extracting text…"):
            text, pages = extract_text_from_pdf(uploaded)

        with st.spinner("✂️ Chunking text…"):
            chunks = chunk_text(text)

        with st.spinner("🔢 Generating embeddings…"):
            embeddings = embed_chunks(chunks)

        with st.spinner("🗄️ Building vector store…"):
            collection = build_chromadb(chunks, embeddings)

        with st.spinner("🔍 Building BM25 index…"):
            bm25 = build_bm25(chunks)

        st.session_state.chunks       = chunks
        st.session_state.collection   = collection
        st.session_state.bm25         = bm25
        st.session_state.pdf_loaded   = True
        st.session_state.pdf_name     = uploaded.name
        st.session_state.page_count   = pages
        st.session_state.chunk_count  = len(chunks)
        st.session_state.messages     = []
        st.session_state.answer_cache = {}
        st.success("✅ PDF ready!")

    if st.session_state.pdf_loaded:
        st.divider()
        st.markdown("**📊 Document Stats**")
        st.metric("File",   st.session_state.pdf_name, delta=None)
        st.metric("Pages",  st.session_state.page_count)
        st.metric("Chunks", st.session_state.chunk_count)
        st.metric("Cached Q&As", len(st.session_state.answer_cache))

        st.divider()
        if st.button("🗑️ Clear chat history"):
            st.session_state.messages = []
            st.rerun()

        if st.button("♻️ Clear answer cache"):
            st.session_state.answer_cache = {}
            st.rerun()

    st.divider()
    st.caption("Stack: Groq · ChromaDB · BM25 · CrossEncoder · SentenceTransformers")


# ─── Main Area ────────────────────────────────────────────────────────────────
st.title("💬 PDF Chatbot")

if not st.session_state.pdf_loaded:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

# Chat history
for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ Latency", f"{meta['latency']}s")
            col2.metric("📦 Chunks used", meta["chunk_count"])
            col3.metric("⚡ Cache hit", "Yes" if meta["from_cache"] else "No")

            with st.expander("🔍 Retrieved source chunks"):
                for i, chunk in enumerate(meta["chunks"]):
                    score = chunk.get("rerank_score", chunk.get("confidence", 0.0))
                    src   = chunk.get("source", "unknown")
                    st.markdown(
                        f"**Chunk {i+1}** &nbsp;|&nbsp; "
                        f"Rerank score: `{score:.4f}` &nbsp;|&nbsp; "
                        f"Source: `{src}`"
                    )
                    st.text_area(
                        label="",
                        value=chunk["text"],
                        height=120,
                        key=f"hist_chunk_{msg_idx}_{i}",
                        disabled=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your PDF…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = get_answer(
                query        = prompt,
                collection   = st.session_state.collection,
                bm25         = st.session_state.bm25,
                chunks       = st.session_state.chunks,
                embed_model  = embed_model,
                reranker     = reranker,
                answer_cache = st.session_state.answer_cache,
            )

        st.markdown(result["answer"])

        col1, col2, col3 = st.columns(3)
        col1.metric("⏱️ Latency",    f"{result['latency']}s")
        col2.metric("📦 Chunks used", len(result["chunks"]))
        col3.metric("⚡ Cache hit",   "Yes" if result["from_cache"] else "No")

        with st.expander("🔍 Retrieved source chunks"):
            for i, chunk in enumerate(result["chunks"]):
                score = chunk.get("rerank_score", chunk.get("confidence", 0.0))
                src   = chunk.get("source", "unknown")
                st.markdown(
                    f"**Chunk {i+1}** &nbsp;|&nbsp; "
                    f"Rerank score: `{score:.4f}` &nbsp;|&nbsp; "
                    f"Source: `{src}`"
                )
                st.text_area(
                    label="",
                    value=chunk["text"],
                    height=120,
                    key=f"new_chunk_{len(st.session_state.messages)}_{i}",
                    disabled=True,
                )

    # Store in history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "meta": {
            "latency":     result["latency"],
            "from_cache":  result["from_cache"],
            "chunk_count": len(result["chunks"]),
            "chunks":      result["chunks"],
        },
    })
