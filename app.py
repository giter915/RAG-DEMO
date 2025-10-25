# -*- coding: utf-8 -*-
"""
DOCX RAG Chatbot — LangChain + FAISS + Streamlit
- Multiple .docx uploads with validation (avoid BadZipFile)
- Chinese-friendly splitting (compatible with multilingual docs)
- Similarity threshold filter + MMR deduplication
- Build FAISS vector index with progress bar and loading indicator
- Session cache (one-time indexing, multi-turn QA)
- Secure configuration reading: environment variable > st.secrets
"""

import os
from typing import List, Optional

import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage

# -----------------------------
# Page configuration & styles
# -----------------------------
st.set_page_config(
    page_title="DOCX RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light UI styling
st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.stChatMessage p {font-size:1.02rem; line-height:1.6;}
section[data-testid="stSidebar"] .st-emotion-cache-1vt4y43 {margin-bottom: 0.5rem;}
code, pre {font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 DOCX RAG Chatbot")
st.caption("LangChain + FAISS + Streamlit (supports multi-doc, MMR, similarity threshold, progress bar)")

# -----------------------------
# Utility functions
# -----------------------------
def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Safely read configs: prefer env vars, fallback to st.secrets, or use default."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default

def is_valid_docx_bytes(b: bytes) -> bool:
    """A valid .docx file starts with 'PK' since it's a zip archive."""
    return b[:2] == b"PK"

def load_docs_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    docs = []
    tmp_dir = "uploaded_docx"
    os.makedirs(tmp_dir, exist_ok=True)
    for uf in uploaded_files:
        raw = uf.read()
        if not is_valid_docx_bytes(raw):
            st.warning(f"⚠️ `{uf.name}` is not a valid .docx (or corrupted). Skipped.")
            continue
        path = os.path.join(tmp_dir, uf.name)
        with open(path, "wb") as f:
            f.write(raw)
        docs.extend(Docx2txtLoader(path).load())
    return docs

def split_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
    )
    return splitter.split_documents(docs)

def build_vectordb_with_progress(splits, embed_model: str, base_url: str, api_key: str) -> FAISS:
    if not splits:
        raise ValueError("No text chunks available for indexing.")

    progress = st.progress(0.0, text="Embedding and building FAISS index… (initializing)")
    status = st.empty()

    embeddings = OpenAIEmbeddings(model=embed_model, base_url=base_url, api_key=api_key)

    total = len(splits)
    batch_size = max(16, min(128, total // 10 or 16))

    first = splits[:batch_size]
    status.info(f"Initializing index… (1/{(total-1)//batch_size + 1} batches)")
    vectordb = FAISS.from_documents(first, embeddings)
    built = len(first)
    progress.progress(min(0.08 + 0.80 * (built/total), 0.95), text=f"Indexed {built}/{total} chunks…")

    while built < total:
        end = min(built + batch_size, total)
        batch = splits[built:end]
        status.info(f"Incremental indexing… ({end//batch_size + (1 if end%batch_size else 0)}/{(total-1)//batch_size + 1} batches)")
        vectordb.add_documents(batch)
        built = end
        progress.progress(min(0.08 + 0.80 * (built/total), 0.98), text=f"Indexed {built}/{total} chunks…")

    progress.progress(1.0, text="Indexing completed ✅")
    progress.empty()
    status.empty()
    return vectordb

def retrieve_docs(vdb: FAISS, query: str, top_k: int, use_mmr: bool, dist_threshold: Optional[float]):
    # Try distance-based filtering first (FAISS scores = distance; smaller is more similar)
    if dist_threshold is not None:
        cands = vdb.similarity_search_with_score(query, k=max(20, top_k * 5))
        picked = []
        for d, dist in cands:
            if dist <= dist_threshold:
                picked.append(d)
            if len(picked) >= top_k:
                break
        if picked:
            return picked

    # Otherwise fallback to MMR or normal retrieval
    if use_mmr:
        retr = vdb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": top_k, "fetch_k": max(10, top_k * 4), "lambda_mult": 0.5},
        )
        return retr.get_relevant_documents(query)
    else:
        retr = vdb.as_retriever(search_kwargs={"k": top_k})
        return retr.get_relevant_documents(query)

def format_context(docs) -> str:
    return "\n\n".join((d.page_content or "").strip() for d in docs)

def make_llm(model: str, base_url: str, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0.2, base_url=base_url, api_key=api_key)

# -----------------------------
# Sidebar configuration
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    default_base_url = read_secret("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    default_api_key  = read_secret("OPENAI_API_KEY", "")
    default_embed    = read_secret("EMBEDDING_MODEL", "text-embedding-3-small")
    default_llm      = read_secret("LLM_MODEL", "gpt-4o-mini")

    base_url   = st.text_input("Base URL (OpenAI compatible)", value=default_base_url, help="e.g. https://www.dmxapi.cn/v1")
    api_key    = st.text_input("API Key (not stored)", type="password", value=default_api_key)
    llm_model  = st.selectbox("Chat Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
                               index=0 if default_llm not in ["gpt-4o", "gpt-4.1-mini"]
                               else ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"].index(default_llm))
    embed_model = st.text_input("Embedding Model", value=default_embed)

    st.divider()
    st.subheader("Retrieval & Chunking")
    top_k = st.slider("Top-K (number of returned chunks)", 1, 10, 4, 1)
    use_mmr = st.checkbox("Enable MMR (Maximal Marginal Relevance)", value=True)
    use_threshold = st.checkbox("Enable similarity threshold (FAISS distance)", value=True)
    dist_threshold = st.slider("Distance threshold (smaller = stricter)", 0.10, 1.00, 0.45, 0.05, disabled=not use_threshold)
    chunk_size = st.number_input("Chunk size", 200, 2000, 800, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 800, 100, 10)

    st.divider()
    clear_btn = st.button("🧹 Clear session and index", use_container_width=True)

# -----------------------------
# Session state
# -----------------------------
if clear_btn:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "ready" not in st.session_state:
    st.session_state.ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "files" not in st.session_state:
    st.session_state.files = []

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = (
    "You are a document-based QA assistant. Always base your answer on the following 'Reference Passages'. "
    "If the reference is insufficient, explicitly state that and suggest what additional info is needed. "
    "Keep answers clear and concise."
)
PROMPT = ChatPromptTemplate.from_template(
    """{sys}

Reference Passages:
{context}

Question:
{question}

Please answer concisely in English. If reference data is insufficient, say so directly."""
)

# -----------------------------
# File upload & vector index building
# -----------------------------
st.subheader("📤 Upload DOCX files (multiple allowed)")
uploaded_files = st.file_uploader("Only .docx supported", type=["docx"], accept_multiple_files=True)

c1, c2 = st.columns([1.2, 2.8])
with c1:
    build_btn = st.button("🚀 Build / Update Index", type="primary", use_container_width=True)
with c2:
    if st.session_state.ready and st.session_state.vectordb is not None:
        st.success(f"Index ready. {len(st.session_state.files)} files loaded in this session.")

if build_btn:
    if not api_key:
        st.error("Please enter your API Key in the sidebar first.")
    elif not uploaded_files:
        st.error("Please upload at least one .docx file.")
    else:
        with st.spinner("Loading and splitting documents…"):
            docs = load_docs_from_uploads(uploaded_files)
            st.session_state.files = [f.name for f in uploaded_files if f is not None]
            if not docs:
                st.error("No valid .docx files were loaded.")
            else:
                splits = split_documents(docs, int(chunk_size), int(chunk_overlap))
                st.info(f"Split into {len(splits)} chunks.")
        try:
            st.session_state.vectordb = build_vectordb_with_progress(
                splits,
                embed_model=embed_model,
                base_url=base_url,
                api_key=api_key,
            )
            st.session_state.ready = True
            st.success("✅ Vector index ready! You can start asking questions.")
        except Exception as e:
            st.exception(e)

# -----------------------------
# Chat area
# -----------------------------
st.subheader("💬 Chat Area")

# Display previous messages
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("Type your question…")
if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            llm = make_llm(llm_model, base_url, api_key)
            use_rag = st.session_state.ready and (st.session_state.vectordb is not None)
            if use_rag:
                hits = retrieve_docs(
                    st.session_state.vectordb,
                    user_q,
                    top_k=int(top_k),
                    use_mmr=bool(use_mmr),
                    dist_threshold=(float(dist_threshold) if use_threshold else None),
                )
                context = format_context(hits) if hits else "(No relevant reference found)"
            else:
                hits = []
                context = "(Index not built, answering without retrieval)"

            msgs = PROMPT.format_messages(sys=SYSTEM_PROMPT, context=context, question=user_q)
            try:
                resp = llm.invoke(msgs)
                answer = resp.content if isinstance(resp, AIMessage) else str(resp)
            except Exception as e:
                answer = f"Model call error: {e}"
            st.markdown(answer)
            st.session_state.messages.append(("assistant", answer))

        # Display sources
        if hits:
            with st.expander("📚 Source documents / matched chunks", expanded=False):
                for i, d in enumerate(hits, 1):
                    src = d.metadata.get("source", "unknown")
                    snippet = (d.page_content or "").strip()
                    st.markdown(f"**[{i}] {src}**")
                    st.code(snippet[:1200])
        else:
            if use_rag:
                st.info("No relevant chunks found. Try lowering the threshold, increasing Top-K, or uploading more relevant docs.")
            else:
                st.info("Tip: Build an index first for more accurate document-grounded answers.")
