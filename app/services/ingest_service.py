import uuid
from pathlib import Path

from app.ingestion.chunk_store import (
    append_chunks_to_store as append_ingestion_chunks_to_store,
)
from app.ingestion.chunk_store import load_chunks_from_store
from app.rag.loader import load_documents
from app.rag.splitter import get_text_splitter
from app.rag.vectorstore import get_vectorstore

def index_document(file_path: Path) -> int:
    docs = load_documents(file_path)
    splitter = get_text_splitter()
    chunks = splitter.split_documents(docs)

    # 给每个 chunk 补 metadata，后续可用于来源追踪
    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = file_path.name
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    append_chunks_to_store(chunks)
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    return len(chunks)

def append_chunks_to_store(chunks):
    append_ingestion_chunks_to_store(chunks)
