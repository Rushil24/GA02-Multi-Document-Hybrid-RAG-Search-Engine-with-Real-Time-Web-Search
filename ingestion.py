import uuid
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, WebBaseLoader
from models import Document
from utils import clean_text

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
        docs = PyPDFLoader(path).load() if suffix == "pdf" else Docx2txtLoader(path).load()
    elif url:
        source_type = "web"
        title = url
        docs = WebBaseLoader(url).load()

    full_text = clean_text(" ".join(d.page_content for d in docs))
    return Document(source_id=str(uuid.uuid4()), source_type=source_type, title=title, content=full_text, metadata={"length": len(full_text)})