from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from models import Document
from utils import is_noise

def index_documents(documents: List[Document]) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=120)
    chunks, metadatas = [], []

    for doc in documents:
        split_texts = splitter.split_text(doc.content)
        for idx, text in enumerate(split_texts):
            if is_noise(text): continue
            chunks.append(text)
            metadatas.append({"source_id": doc.source_id, "source_type": doc.source_type, "title": doc.title, "chunk_index": idx})

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)