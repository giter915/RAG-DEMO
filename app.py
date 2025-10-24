# -*- coding: utf-8 -*-
"""
DOCX RAG Chatbot — LangChain + FAISS + Streamlit
- 多文件上传（.docx），有效性校验（防 BadZipFile）
- 中文友好切分
- 相似度阈值过滤 + MMR 去冗余
- 构建向量索引带进度条、等待提示
- 会话缓存（一次索引，多轮问答）
- 安全读取配置：优先环境变量，再尝试 st.secrets
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
# 页面基础配置与样式
# -----------------------------
st.set_page_config(
    page_title="DOCX RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 轻度美化
st.markdown("""
<style>
/* 顶部标题缩小留白 */
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
/* 聊天消息内容更易读 */
.stChatMessage p {font-size:1.02rem; line-height:1.6;}
/* 侧栏标题间距 */
section[data-testid="stSidebar"] .st-emotion-cache-1vt4y43 {margin-bottom: 0.5rem;}
/* 代码块更紧凑 */
code, pre {font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 DOCX RAG Chatbot")
st.caption("LangChain + FAISS + Streamlit（支持多文档、MMR、相似度阈值、进度条）")

# -----------------------------
# 工具函数
# -----------------------------
def read_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """安全读取配置：环境变量优先，再尝试 st.secrets；都无则给默认值"""
    v = os.environ.get(name)
    if v:
        return v
    try:
        return st.secrets[name]
    except Exception:
        return default

def is_valid_docx_bytes(b: bytes) -> bool:
    """docx 本质是 zip，前两个字节为 'PK'"""
    return b[:2] == b"PK"

def load_docs_from_uploads(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]):
    docs = []
    tmp_dir = "uploaded_docx"
    os.makedirs(tmp_dir, exist_ok=True)
    for uf in uploaded_files:
        raw = uf.read()
        if not is_valid_docx_bytes(raw):
            st.warning(f"⚠️ `{uf.name}` 不是有效的 .docx（或已损坏），已跳过。")
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
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""],
    )
    return splitter.split_documents(docs)

def build_vectordb_with_progress(splits, embed_model: str, base_url: str, api_key: str) -> FAISS:
    if not splits:
        raise ValueError("没有可索引的文本片段。")

    progress = st.progress(0.0, text="正在向量化与建立索引…（启动中）")
    status = st.empty()

    embeddings = OpenAIEmbeddings(model=embed_model, base_url=base_url, api_key=api_key)

    total = len(splits)
    batch_size = max(16, min(128, total // 10 or 16))

    first = splits[:batch_size]
    status.info(f"索引初始化…（1/{(total-1)//batch_size + 1} 批）")
    vectordb = FAISS.from_documents(first, embeddings)
    built = len(first)
    progress.progress(min(0.08 + 0.80 * (built/total), 0.95), text=f"已建立 {built}/{total} 个片段…")

    while built < total:
        end = min(built + batch_size, total)
        batch = splits[built:end]
        status.info(f"增量索引…（{end//batch_size + (1 if end%batch_size else 0)}/{(total-1)//batch_size + 1} 批）")
        # ✅ 修复：新版 LangChain 不再需要传 embeddings
        vectordb.add_documents(batch)
        built = end
        progress.progress(min(0.08 + 0.80 * (built/total), 0.98), text=f"已建立 {built}/{total} 个片段…")

    progress.progress(1.0, text="索引完成 ✅")
    progress.empty()
    status.empty()
    return vectordb

def retrieve_docs(vdb: FAISS, query: str, top_k: int, use_mmr: bool, dist_threshold: Optional[float]):
    # 先尝试“距离阈值”过滤（FAISS 分数是距离：越小越相似）
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

    # 否则退回 MMR 或普通检索
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
# 侧边栏（确保显示）
# -----------------------------
with st.sidebar:
    st.header("⚙️ 设置")

    default_base_url = read_secret("OPENAI_BASE_URL", "https://www.dmxapi.cn/v1")
    default_api_key  = read_secret("OPENAI_API_KEY", "")
    default_embed    = read_secret("EMBEDDING_MODEL", "text-embedding-3-small")
    default_llm      = read_secret("LLM_MODEL", "gpt-4o-mini")

    base_url   = st.text_input("Base URL（OpenAI 兼容）", value=default_base_url, help="例如：https://www.dmxapi.cn/v1")
    api_key    = st.text_input("API Key（不会保存）", type="password", value=default_api_key)
    llm_model  = st.selectbox("对话模型", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
                               index=0 if default_llm not in ["gpt-4o", "gpt-4.1-mini"]
                               else ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"].index(default_llm))
    embed_model = st.text_input("Embedding 模型", value=default_embed)

    st.divider()
    st.subheader("检索与切分")
    top_k = st.slider("Top-K（返回片段数）", 1, 10, 4, 1)
    use_mmr = st.checkbox("启用 MMR 去冗余检索", value=True)
    use_threshold = st.checkbox("启用相似度阈值过滤（基于 FAISS 距离）", value=True)
    dist_threshold = st.slider("距离阈值（越小越严格）", 0.10, 1.00, 0.45, 0.05, disabled=not use_threshold)
    chunk_size = st.number_input("切片大小 chunk_size", 200, 2000, 800, 50)
    chunk_overlap = st.number_input("切片重叠 chunk_overlap", 0, 800, 100, 10)

    st.divider()
    clear_btn = st.button("🧹 清空会话与索引", use_container_width=True)

# -----------------------------
# 会话状态
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
# 系统 Prompt
# -----------------------------
SYSTEM_PROMPT = (
    "你是一个文档问答助手。请优先基于下方“资料片段”回答；"
    "若资料不足，请如实说明并给出需要补充的内容。回答条理清晰、简洁。"
)
PROMPT = ChatPromptTemplate.from_template(
    """{sys}

资料片段：
{context}

问题：
{question}

请用中文简洁作答；若资料不足请直说。"""
)

# -----------------------------
# 上传与构建索引
# -----------------------------
st.subheader("📤 上传 DOCX 文档（可多选）")
uploaded_files = st.file_uploader("仅支持 .docx", type=["docx"], accept_multiple_files=True)

c1, c2 = st.columns([1.2, 2.8])
with c1:
    build_btn = st.button("🚀 构建 / 更新索引", type="primary", use_container_width=True)
with c2:
    if st.session_state.ready and st.session_state.vectordb is not None:
        st.success(f"索引已就绪。本次会话共载入 {len(st.session_state.files)} 个文件。")

if build_btn:
    if not api_key:
        st.error("请先在侧边栏输入 API Key。")
    elif not uploaded_files:
        st.error("请先上传至少一个 .docx 文件。")
    else:
        with st.spinner("正在加载与切分文档…"):
            docs = load_docs_from_uploads(uploaded_files)
            st.session_state.files = [f.name for f in uploaded_files if f is not None]
            if not docs:
                st.error("没有成功加载的 .docx 文档。")
            else:
                splits = split_documents(docs, int(chunk_size), int(chunk_overlap))
                st.info(f"已切分为 {len(splits)} 个片段。")
        try:
            st.session_state.vectordb = build_vectordb_with_progress(
                splits,
                embed_model=embed_model,
                base_url=base_url,
                api_key=api_key,
            )
            st.session_state.ready = True
            st.success("✅ 向量索引就绪！可以开始提问。")
        except Exception as e:
            st.exception(e)

# -----------------------------
# 聊天区
# -----------------------------
st.subheader("💬 提问区")

# 展示历史消息
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

user_q = st.chat_input("输入你的问题…")
if user_q:
    st.session_state.messages.append(("user", user_q))
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
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
                context = format_context(hits) if hits else "(未检索到相关资料)"
            else:
                hits = []
                context = "(未构建索引，本次为无检索回答)"

            msgs = PROMPT.format_messages(sys=SYSTEM_PROMPT, context=context, question=user_q)
            try:
                resp = llm.invoke(msgs)
                answer = resp.content if isinstance(resp, AIMessage) else str(resp)
            except Exception as e:
                answer = f"模型调用出错：{e}"
            st.markdown(answer)
            st.session_state.messages.append(("assistant", answer))

        # 展示来源
        if hits:
            with st.expander("📚 来源文档 / 命中片段", expanded=False):
                for i, d in enumerate(hits, 1):
                    src = d.metadata.get("source", "unknown")
                    snippet = (d.page_content or "").strip()
                    st.markdown(f"**[{i}] {src}**")
                    st.code(snippet[:1200])
        else:
            if use_rag:
                st.info("未检索到相关片段，请尝试降低阈值、增大 Top-K，或上传更相关的文档。")
            else:
                st.info("提示：构建索引后可获得更可靠的基于资料的回答。")

