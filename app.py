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

# ─── API KEYS ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.environ.get("TAVILY_API_KEY", ""))
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
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_text_from_pdf(uploaded_file) -> tuple[str, int]:
    """Extract all text from an uploaded PDF. Returns (text, page_count)."""
    reader = PyPDF2.PdfReader(uploaded_file)
    pages = []
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
    q_emb   = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs    = results["documents"][0]
    return [{"text": d, "source": "semantic"} for d in docs]

def bm25_retrieve(query: str, bm25, chunks: list[str], top_k: int = 6) -> list[dict]:
    """BM25 keyword retrieval."""
    scores  = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{"text": chunks[i], "bm25_score": float(scores[i]), "source": "bm25"} for i in top_idx]

def hybrid_search(query: str, collection, bm25, chunks: list[str],
                  embed_model, top_k: int = 10) -> list[dict]:
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
                         embed_model, reranker, top_k: int = 5) -> list[dict]:
    """Hybrid search → CrossEncoder reranking → top_k results."""
    candidates = hybrid_search(query, collection, bm25, chunks, embed_model, top_k=10)
    pairs      = [[query, c["text"]] for c in candidates]
    scores     = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_k]

def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the LLM prompt. LLM is asked to self-classify its answer mode."""
    context_parts = [c["text"] for c in chunks]
    context = "\n\n---\n\n".join(context_parts)
    return f"""You are a strict document QA assistant.
Read ALL context passages carefully before answering.

Your response MUST start with exactly one of these prefixes on its own line:
  DIRECT:     — one passage clearly and specifically answers the question
  ANALYTICAL: — the specific answer requires combining information across multiple passages
  NOTFOUND:   — the passages do not specifically answer the question asked.
                Use this even if the passages contain content on a related topic.
                Example: if asked about "data migration objects" but the document only
                lists "key components", that is NOTFOUND — not the same thing.

Then on the next line, write your answer.

Rules:
- Answer only from the provided context. Never invent facts.
- Do not reference passage numbers or say "according to passage X".
- If using NOTFOUND, write one sentence explaining what the document covers instead.
- Be thorough — if the answer spans multiple passages, include all relevant details.

CONTEXT:
{context}

QUESTION: {query}

RESPONSE:"""


def _parse_llm_response(raw: str) -> tuple[str, str]:
    """
    Parse the LLM's prefixed response.
    Returns (answer_type, clean_answer).
    answer_type is one of: 'direct', 'analytical', 'notfound'
    Falls back to 'direct' if prefix is missing or unrecognised.
    """
    raw = raw.strip()
    for prefix, atype in [
        ("DIRECT:",     "direct"),
        ("ANALYTICAL:", "analytical"),
        ("NOTFOUND:",   "notfound"),
    ]:
        if raw.upper().startswith(prefix):
            clean = raw[len(prefix):].strip()
            return atype, clean
    # No recognised prefix — treat as direct, return as-is
    return "direct", raw


def web_search_fallback(query: str) -> str:
    """
    Search the web using Tavily (built for AI/RAG apps) and summarise
    results via Groq.  Falls back to LLM general knowledge if Tavily
    is unavailable or returns no results.
    """
    try:
        from tavily import TavilyClient

        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not set")

        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=5)
        hits    = results.get("results", [])

        if not hits:
            raise ValueError("Tavily returned no results")

        web_context = ""
        sources     = []
        for i, r in enumerate(hits[:5]):
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            web_context += f"[Source {i+1}] {title}\n{content}\n\n"
            if url:
                sources.append(f"- {title}: {url}")

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Using the web search results provided, write a clean, well-synthesised answer "
                        "in 3-5 sentences. "
                        "Do NOT say 'Source 1 says', 'According to Source 2', or reference sources by number. "
                        "Just write the answer directly as a single coherent paragraph. "
                        "Do not invent facts beyond what the sources say."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Web search results:\n{web_context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )

        answer      = response.choices[0].message.content.strip()
        source_text = "\n".join(sources[:3])
        answer += (
            f"\n\n---\n🌐 **Answered from web search** — not found in the uploaded document."
            f"\n\n**Sources:**\n{source_text}"
        )
        return answer

    except Exception:
        # Final fallback: LLM general knowledge
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer clearly and concisely in 3-5 sentences using your general knowledge.",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            answer  = response.choices[0].message.content.strip()
            answer += "\n\n---\n💡 **Answered from general knowledge** — not found in the uploaded document."
            return answer
        except Exception:
            return None


def _is_garbled(chunks: list[dict]) -> bool:
    """
    Return True only if chunks are BOTH heavily non-ASCII AND have deeply
    negative rerank scores. This prevents false positives on academic PDFs
    (like the RAG paper) that have some math notation but still contain
    readable answers.

    FIX: Changed from (ratio > 0.02 OR score < 0.1) to
         (ratio > 0.15 AND score < -5.0)
    The old OR logic was too aggressive — a rerank score of -0.04 (which is
    actually decent) was incorrectly triggering the garbled fallback.
    """
    if not chunks:
        return True
    top_text  = chunks[0].get("text", "")
    non_ascii = sum(1 for c in top_text if ord(c) > 127)
    ratio     = non_ascii / max(len(top_text), 1)
    top_score = chunks[0].get("rerank_score", 1.0)
    # FIXED LINE — both conditions must be true, with much looser thresholds
    return ratio > 0.15 and top_score < -5.0

def web_search_fallback(query: str) -> str:
    """
    Search the web using Tavily (built for AI/RAG apps) and summarise
    results via Groq.  Falls back to LLM general knowledge if Tavily
    is unavailable or returns no results.
    """
    try:
        from tavily import TavilyClient

        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not set")

        client  = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=5)
        hits    = results.get("results", [])

        if not hits:
            raise ValueError("Tavily returned no results")

        web_context = ""
        sources     = []
        for i, r in enumerate(hits[:5]):
            title   = r.get("title", "")
            content = r.get("content", "")
            url     = r.get("url", "")
            web_context += f"[Source {i+1}] {title}\n{content}\n\n"
            if url:
                sources.append(f"- {title}: {url}")

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Using the web search results provided, write a clean, well-synthesised answer "
                        "in 3-5 sentences. "
                        "Do NOT say 'Source 1 says', 'According to Source 2', or reference sources by number. "
                        "Just write the answer directly as a single coherent paragraph. "
                        "Do not invent facts beyond what the sources say."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Web search results:\n{web_context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ],
            temperature=0.0,
            max_tokens=400,
        )

        answer      = response.choices[0].message.content.strip()
        source_text = "\n".join(sources[:3])
        answer += (
            f"\n\n---\n🌐 **Answered from web search** — not found in the uploaded document."
            f"\n\n**Sources:**\n{source_text}"
        )
        return answer

    except Exception:
        # Final fallback: LLM general knowledge
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer clearly and concisely in 3-5 sentences using your general knowledge.",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            answer  = response.choices[0].message.content.strip()
            answer += "\n\n---\n💡 **Answered from general knowledge** — not found in the uploaded document."
            return answer
        except Exception:
            return None

def get_answer(query: str, collection, bm25, chunks: list[str],
               embed_model, reranker, answer_cache: dict) -> dict:
    """Full pipeline: cache check → hybrid retrieve → rerank → LLM answer."""
    cache_key = hashlib.md5(query.strip().lower().encode()).hexdigest()
    if cache_key in answer_cache:
        cached = dict(answer_cache[cache_key])
        cached["from_cache"] = True
        return cached

    t0        = time.time()
    retrieved = full_hybrid_retrieve(query, collection, bm25, chunks, embed_model, reranker)

    # ── Garbled chunk detection ───────────────────────────────────────────────
    if _is_garbled(retrieved):
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer clearly and concisely "
                        "in 2-4 sentences. Give a direct, accurate answer."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        answer      = response.choices[0].message.content.strip()
        answer     += "\n\n*(Note: answered from general knowledge — the relevant document section contained mathematical notation that could not be parsed.)*"
        answer_type = "general"

    else:
        # ── Normal path: answer from document context ─────────────────────────
        prompt   = build_prompt(query, retrieved)
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict document QA assistant. "
                        "Always begin your response with DIRECT:, ANALYTICAL:, or NOTFOUND: "
                        "as instructed in the user prompt. Never skip this prefix."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        raw = response.choices[0].message.content.strip()
        answer_type, answer = _parse_llm_response(raw)

        # ── Web search fallback if LLM says not found ─────────────────────────
        if answer_type == "notfound":
            web_answer = web_search_fallback(query)
            if web_answer:
                answer      = web_answer
                answer_type = "web"

    latency = round(time.time() - t0, 2)

    result  = {
        "answer":      answer,
        "chunks":      retrieved,
        "latency":     latency,
        "from_cache":  False,
        "answer_type": answer_type,
    }
    answer_cache[cache_key] = result
    return result

# ─── Answer Type Badge ───────────────────────────────────────────────────────
def _answer_type_badge(answer_type: str) -> str:
    """Return a markdown badge string for the answer type."""
    badges = {
        "direct":     "📄 **Answered directly from document**",
        "analytical": "🧩 **Synthesised across multiple passages**",
        "web":        "🌐 **Answered from web search**",
        "general":    "💡 **Answered from general knowledge**",
        "cached":     "⚡ **Returned from cache**",
    }
    return badges.get(answer_type, "")

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

        st.session_state.chunks      = chunks
        st.session_state.collection  = collection
        st.session_state.bm25        = bm25
        st.session_state.pdf_loaded  = True
        st.session_state.pdf_name    = uploaded.name
        st.session_state.page_count  = pages
        st.session_state.chunk_count = len(chunks)
        st.session_state.messages    = []
        st.session_state.answer_cache = {}
        st.success("✅ PDF ready!")

    if st.session_state.pdf_loaded:
        st.divider()
        st.markdown("**📊 Document Stats**")
        st.metric("File",        st.session_state.pdf_name,  delta=None)
        st.metric("Pages",       st.session_state.page_count)
        st.metric("Chunks",      st.session_state.chunk_count)
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
            # ── Answer type badge ─────────────────────────────────────────
            atype = meta.get("answer_type", "")
            if meta.get("from_cache"):
                atype = "cached"
            badge = _answer_type_badge(atype)
            if badge and "🌐" not in msg["content"] and "💡" not in msg["content"]:
                st.caption(badge)
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ Latency",    f"{meta['latency']}s")
            col2.metric("📦 Chunks used", meta["chunk_count"])
            col3.metric("⚡ Cache hit",   "Yes" if meta["from_cache"] else "No")
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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
        # ── Answer type badge ─────────────────────────────────────────────────
        atype = result.get("answer_type", "")
        if result.get("from_cache"):
            atype = "cached"
        badge = _answer_type_badge(atype)
        if badge and "🌐" not in result["answer"] and "💡" not in result["answer"]:
            st.caption(badge)
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

    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "meta": {
            "latency":     result["latency"],
            "from_cache":  result["from_cache"],
            "chunk_count": len(result["chunks"]),
            "chunks":      result["chunks"],
            "answer_type": result.get("answer_type", ""),
        },
    })
