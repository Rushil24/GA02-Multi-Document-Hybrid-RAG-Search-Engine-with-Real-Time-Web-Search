# GA02 – Multi-Document Hybrid RAG Search Engine with Real-Time Web Search

## 📌 Overview

This project implements a **Hybrid Retrieval-Augmented Generation (RAG) Search Engine** that intelligently combines **semantic search over local documents** with **real-time web search** to answer user queries in a grounded, explainable, and citation-aware manner.

The system is designed to mirror **enterprise research assistants and internal AI copilots**, where users query both **private knowledge bases** (PDFs, documents) and **live internet data** while avoiding hallucination.

---

## 🎯 Problem Statement

Organizations store knowledge across multiple unstructured sources such as PDFs, reports, notes, and reference documents. Traditional keyword search fails to capture semantic meaning, while LLMs alone may hallucinate answers or lack real-time awareness.

This project addresses the following challenges:

* Semantic search across multiple documents
* Combining internal document knowledge with live web data
* Preventing hallucination through context grounding
* Providing transparent answers with clear citations
* Building an end-user friendly chatbot interface

---

## 🧠 Key Features

* 📄 **Multi-document ingestion** (PDF, DOCX, URLs)
* 🔍 **Semantic search using FAISS**
* 🔀 **Hybrid RAG pipeline** (Documents + Web)
* 🌐 **Real-time web search using Tavily**
* 🧭 **Query routing logic** (Document / Web / Hybrid)
* 📝 **Citation-aware answer generation**
* 💬 **Interactive Streamlit chatbot UI**
* ⚙️ **Windows-compatible & CPU-friendly setup**

---

## 🏗️ System Architecture

The system consists of five major components:

1. **Document Ingestion**

   * Loads PDFs, DOCX files, and web pages using LangChain loaders
   * Cleans and normalizes text
   * Maintains consistent metadata for traceability

2. **Chunking & Embedding**

   * Recursive character-based chunking with overlap
   * Sentence Transformers used for embeddings
   * Metadata preserved per chunk

3. **Vector Database**

   * FAISS used for fast semantic similarity search
   * Stores document embeddings and metadata

4. **Hybrid Retrieval Layer**

   * Query classification into document, web, or hybrid
   * Tavily used for real-time web search
   * Web results treated as temporary context

5. **Answer Generation & UI**

   * Flan-T5 used for instruction-based generation
   * Answers generated strictly from retrieved context
   * Clear source attribution and citations
   * Streamlit-based chat interface

---

## 🧪 Query Routing Examples

| Query                       | Routing Type |
| --------------------------- | ------------ |
| Explain attention mechanism | 📄 Document  |
| Latest AI news today        | 🌐 Web       |
| Who is Narendra Modi        | 🔀 Hybrid    |

This routing prevents hallucination and ensures relevant data sources are used.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **LLM Orchestration:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** Sentence Transformers (MiniLM)
* **LLM:** Flan-T5 (Transformers)
* **Web Search:** Tavily API
* **Frontend:** Streamlit

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <your-github-repo-url>
cd <repo-folder>
```

### 2️⃣ Create & Activate Environment

```bash
conda create -n ga03 python=3.10
conda activate ga03
```

### 3️⃣ Install Dependencies

### 4️⃣ Set Tavily API Key (Required for Web Search)

```bash
setx TAVILY_API_KEY "your_api_key_here"
```

Restart the terminal after setting the key.

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

Open the browser at:

```
http://localhost:8501
```

---

## 🔐 Tavily API Key Setup

1. Visit: [https://tavily.com](https://tavily.com)
2. Sign up / log in
3. Generate API key from dashboard
4. Store it securely as an environment variable

⚠️ Do not hardcode API keys or push them to GitHub.

---

## 📊 Evaluation Summary

### Strengths

* Prevents hallucination by grounding answers in context
* Seamlessly integrates document and web data
* Clear citations for transparency
* Modular and extensible design

### Limitations

* CPU-only inference may be slower
* Web search depends on API availability

### Future Enhancements

* Persistent FAISS index storage
* Chat history & memory
* Improved query classification
* Cloud deployment (Docker / AWS)

---

## 🎓 Learning Outcomes

This project demonstrates:

* Multi-document RAG system design
* Hybrid retrieval strategies (Vector + Web)
* Real-time data integration
* Citation-aware answer generation
* Practical LangChain + Streamlit application development
* Real-world debugging and system design skills

---

## 📌 Conclusion

This Hybrid RAG Search Engine goes beyond basic AI demos and reflects **real-world enterprise AI system design**. By combining semantic search, live web retrieval, controlled generation, and explainability, it closely mirrors how modern research assistants and AI copilots are built in practice.
