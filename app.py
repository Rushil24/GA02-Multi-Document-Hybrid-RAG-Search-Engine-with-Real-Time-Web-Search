import streamlit as st
from ingestion import load_document
from indexing import index_documents
from ai_engine import answer_question

st.set_page_config("GA02 Hybrid RAG", layout="wide")
st.title("🔍 GA02 – Multi-Document Hybrid RAG Search Engine")

if "documents" not in st.session_state: st.session_state.documents = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None

# Sidebar logic
with st.sidebar:
    st.header("📄 Management")
    uploaded = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])
    url = st.text_input("Or URL")
    st.checkbox("🌐 Enable Tavily", key="use_web", value=True)
    if st.button("Ingest"):
        st.session_state.documents.append(load_document(uploaded, url))
    if st.button("Build Index"):
        st.session_state.vectorstore = index_documents(st.session_state.documents)

# Main Chat
question = st.text_input("Ask a research question")
if question and st.session_state.vectorstore:
    ans, src, qtype = answer_question(question, st.session_state.vectorstore)
    st.write(ans)
    with st.expander("Sources"):
        for s in src: st.write(f"[{s.source_type}] {s.reference}")