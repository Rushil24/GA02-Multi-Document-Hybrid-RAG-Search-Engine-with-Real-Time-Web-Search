# ============================================================
# GA02 – Multi-Document Hybrid RAG Search Engine
# Documents + FAISS + Tavily Web Search
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import uuid
import re
import tempfile
from typing import List, Dict
from dataclasses import dataclass

# ---------------- LangChain ----------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader
)
from langchain.tools.tavily_search import TavilySearchResults

# ---------------- LLM ----------------
from transformers.pipelines import pipeline

# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Document:
    source_id: str
    source_type: str      # pdf | web
    title: str
    content: str
    metadata: Dict

@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str

@dataclass
class AnswerSource:
    source_type: str      # doc | web
    reference: str

# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_noise(text: str) -> bool:
    noise_terms = [
        "references", "acknowledgment", "biography",
        "editor", "committee", "©", "ieee"
    ]
    t = text.lower()
    return any(n in t for n in noise_terms)

# ============================================================
# DOCUMENT INGESTION
# ============================================================

def load_document(uploaded_file=None, url=None) -> Document:
    docs = []
    source_type = "pdf"
    title = ""

    if uploaded_file:
        suffix = uploaded_file.name.split(".")[-1]
        title = uploaded_file.name

        with tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix) as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        if suffix == "pdf":
            docs = PyPDFLoader(path).load()
        else:
            docs = Docx2txtLoader(path).load()

    elif url:
        source_type = "web"
        title = url
        docs = WebBaseLoader(url).load()

    full_text = clean_text(" ".join(d.page_content for d in docs))

    return Document(
        source_id=str(uuid.uuid4()),
        source_type=source_type,
        title=title,
        content=full_text,
        metadata={"length": len(full_text)}
    )

# ============================================================
# VECTOR INDEXING
# ============================================================

def index_documents(documents: List[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=120
    )

    chunks = []
    metadatas = []

    for doc in documents:
        split_texts = splitter.split_text(doc.content)

        for idx, text in enumerate(split_texts):
            if is_noise(text):
                continue

            chunks.append(text)
            metadatas.append({
                "source_id": doc.source_id,
                "source_type": doc.source_type,
                "title": doc.title,
                "chunk_index": idx
            })

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

# ============================================================
# QUERY CLASSIFICATION
# ============================================================

def classify_query(query: str) -> str:
    q = query.lower()

    realtime_keywords = [
        "latest", "recent", "current",
        "news", "today", "now",
        "2024", "2025"
    ]

    general_knowledge_patterns = [
        "who is", "what is", "define",
        "biography", "born", "about"
    ]

    if any(k in q for k in realtime_keywords):
        return "web"

    if any(p in q for p in general_knowledge_patterns):
        return "hybrid"

    if "compare" in q:
        return "hybrid"

    return "document"


# ============================================================
# TAVILY WEB SEARCH
# ============================================================

tavily_tool = TavilySearchResults(k=5)

def tavily_search(query: str) -> List[WebSearchResult]:
    results = tavily_tool.run(query)

    web_results = []
    for r in results:
        web_results.append(
            WebSearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")
            )
        )

    return web_results

# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.2
    )

# ============================================================
# HYBRID CONTEXT ASSEMBLY
# ============================================================

def build_context(query, vectorstore, allow_web=True):
    query_type = classify_query(query)

    contexts = []
    sources = []

    if query_type in ["document", "hybrid"]:
        docs = vectorstore.similarity_search(query, k=5)
        for d in docs:
            contexts.append(d.page_content)
            meta = d.metadata
            sources.append(
                AnswerSource(
                    source_type="doc",
                    reference=f"{meta['title']} – Chunk{meta['chunk_index']}"
                )
            )

    if query_type in ["web", "hybrid"] and allow_web:
        web_results = tavily_search(query)
        for w in web_results:
            contexts.append(w.snippet)
            sources.append(
                AnswerSource(
                    source_type="web",
                    reference=f"Tavily – {w.title}"
                )
            )

    return "\n\n".join(contexts[:8]), sources, query_type

# ============================================================
# RAG QA
# ============================================================

def answer_question(question, vectorstore):
    context, sources, qtype = build_context(
        question,
        vectorstore,
        allow_web=st.session_state.get("use_web", True)
    )

    prompt = f"""
You are an academic research assistant.

Answer ONLY using the context below.
If the answer is missing, say so clearly.

Context:
{context}

Question:
{question}

Answer:
"""

    llm = load_llm()
    answer = llm(prompt)[0]["generated_text"]

    return answer.strip(), sources, qtype

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config("GA02 Hybrid RAG", layout="wide")
st.title("🔍 GA02 – Multi-Document Hybrid RAG Search Engine")

if "documents" not in st.session_state:
    st.session_state.documents: List[Document] = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------------- SIDEBAR ----------------
st.sidebar.header("📄 Document Management")

uploaded = st.sidebar.file_uploader(
    "Upload PDF or DOCX", type=["pdf", "docx"]
)

url = st.sidebar.text_input("Or paste document URL")

st.sidebar.checkbox("🌐 Enable Tavily Web Search", key="use_web", value=True)

if st.sidebar.button("Ingest Document"):
    doc = load_document(uploaded, url)
    st.session_state.documents.append(doc)
    st.success("Document ingested successfully")

if st.sidebar.button("Build Knowledge Index"):
    st.session_state.vectorstore = index_documents(
        st.session_state.documents
    )
    st.success("FAISS knowledge index built")

# ---------------- DOCUMENT LIST ----------------
st.header("📚 Indexed Documents")

for d in st.session_state.documents:
    with st.expander(d.title):
        st.write(f"Source Type: {d.source_type}")
        st.write(d.content[:1000] + "...")

# ---------------- HYBRID CHAT ----------------
st.header("💬 Hybrid Research Chat")

question = st.text_input("Ask a research question")

if question and st.session_state.vectorstore:
    answer, sources, qtype = answer_question(
        question,
        st.session_state.vectorstore
    )

    icon = {"document": "📄", "web": "🌐", "hybrid": "🔀"}[qtype]
    st.subheader(f"{icon} Answer")
    st.write(answer)

    with st.expander("📑 Evidence & Sources"):
        for s in sources:
            st.write(f"[{s.source_type.upper()}] {s.reference}")
