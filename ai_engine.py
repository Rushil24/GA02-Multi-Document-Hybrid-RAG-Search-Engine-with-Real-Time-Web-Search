import streamlit as st
from transformers.pipelines import pipeline
from langchain_tools.tavily_search import TavilySearchResults
from models import WebSearchResult, AnswerSource

tavily_tool = TavilySearchResults(k=5)

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, temperature=0.2)

def classify_query(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["latest", "recent", "news", "2024", "2025"]): return "web"
    if any(p in q for p in ["who is", "what is", "compare"]): return "hybrid"
    return "document"

def tavily_search_logic(query: str):
    results = tavily_tool.run(query)
    return [WebSearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", "")) for r in results]

def answer_question(question, vectorstore):
    qtype = classify_query(question)
    contexts, sources = [], []

    if qtype in ["document", "hybrid"]:
        docs = vectorstore.similarity_search(question, k=5)
        for d in docs:
            contexts.append(d.page_content)
            sources.append(AnswerSource(source_type="doc", reference=f"{d.metadata['title']}"))

    if qtype in ["web", "hybrid"] and st.session_state.get("use_web", True):
        web_res = tavily_search_logic(question)
        for w in web_res:
            contexts.append(w.snippet)
            sources.append(AnswerSource(source_type="web", reference=f"Tavily - {w.title}"))

    prompt = f"Context:\n{chr(10).join(contexts[:8])}\n\nQuestion:\n{question}\n\nAnswer:"
    answer = load_llm()(prompt)[0]["generated_text"]
    return answer.strip(), sources, qtype