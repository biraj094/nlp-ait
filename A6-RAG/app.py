import streamlit as st
import os
import sqlite3
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

st.set_page_config(
    page_title="Chapter 11 – Contextual RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Chapter 11: Contextual Retrieval Chatbot")
st.caption("Powered by pre-built contextual embeddings from A6 notebook")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password",
                                  placeholder="gsk_…")
    st.markdown("---")
    st.info(
        "**Pipeline**\n"
        "1. Reads docs from `chroma_contextual/chroma.sqlite3`\n"
        "2. Builds FAISS index (first run only)\n"
        "3. Retrieve top-3 enriched chunks\n"
        "4. Generate answer via LLaMA-3.1-8B (Groq)"
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if not groq_api_key:
    st.warning("Enter your Groq API key in the sidebar to begin.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

SQLITE_PATH = "./chroma_contextual/chroma.sqlite3"
FAISS_DIR   = "./faiss_contextual"

if not os.path.exists(SQLITE_PATH):
    st.error(f"`{SQLITE_PATH}` not found. Make sure `chroma_contextual/` is next to `app.py`.")
    st.stop()


@st.cache_resource
def get_embeddings():
    print("[INFO] Loading FastEmbed model...", flush=True)
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("[INFO] FastEmbed model loaded.", flush=True)
    return emb

@st.cache_resource
def get_llm(_api_key: str):
    print("[INFO] Initialising Groq LLM...", flush=True)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, groq_api_key=_api_key)
    print("[INFO] Groq LLM ready.", flush=True)
    return llm

def read_docs_from_sqlite(sqlite_path: str) -> list:
    print(f"[INFO] Reading docs from {sqlite_path}...", flush=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, c0 as document
        FROM embedding_fulltext_search_content
        ORDER BY id
    """).fetchall()

    meta_rows = conn.execute("""
        SELECT id, key, string_value, int_value, float_value
        FROM embedding_metadata
    """).fetchall()
    conn.close()

    meta_map = {}
    for m in meta_rows:
        eid = m["id"]
        if eid not in meta_map:
            meta_map[eid] = {}
        val = m["string_value"] or m["int_value"] or m["float_value"]
        meta_map[eid][m["key"]] = val

    docs = []
    for row in rows:
        text = row["document"]
        if text:
            metadata = meta_map.get(row["id"], {})
            docs.append(Document(page_content=text, metadata=metadata))

    print(f"[INFO] Loaded {len(docs)} docs from SQLite.", flush=True)
    return docs


@st.cache_resource(show_spinner=False)
def load_faiss_db():
    emb = get_embeddings()

    if os.path.exists(FAISS_DIR):
        print("[INFO] Loading existing FAISS index...", flush=True)
        db = FAISS.load_local(FAISS_DIR, emb, allow_dangerous_deserialization=True)
        print(f"[INFO] FAISS loaded: {db.index.ntotal} vectors.", flush=True)
        return db, db.index.ntotal, "cache"

    print("[INFO] Building FAISS index from SQLite...", flush=True)
    docs = read_docs_from_sqlite(SQLITE_PATH)
    db = FAISS.from_documents(docs, emb)
    db.save_local(FAISS_DIR)
    print(f"[INFO] FAISS built and saved: {len(docs)} docs.", flush=True)
    return db, len(docs), "built"

with st.status("⚡ Loading vector store…", expanded=True) as status:
    vector_db, n_chunks, source = load_faiss_db()
    if source == "built":
        status.update(
            label=f"✅ Built FAISS index from {n_chunks} chunks (saved to `{FAISS_DIR}`)",
            state="complete", expanded=False,
        )
    else:
        status.update(
            label=f"✅ Loaded {n_chunks} chunks from FAISS index",
            state="complete", expanded=False,
        )

llm = get_llm(groq_api_key)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📍 Source Chunks (Enriched)"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}** — page {src['page']}")
                    st.info(src["content"])
                    st.divider()

if user_input := st.chat_input("Ask a question about Chapter 11…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating…"):
            response = qa_chain.invoke({"query": user_input})

        answer = response["result"]
        source_docs = response["source_documents"]

        st.markdown(answer)

        sources_meta = []
        with st.expander("📍 Source Chunks (Enriched)"):
            for i, doc in enumerate(source_docs, 1):
                page = doc.metadata.get("page", "?")
                st.markdown(f"**Source {i}** — page {page}")
                st.info(doc.page_content)
                st.divider()
                sources_meta.append({"page": page, "content": doc.page_content})

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources_meta,
    })
