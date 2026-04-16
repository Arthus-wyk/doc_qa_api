import json
import uuid
from pathlib import Path
from app.rag.loader import load_documents
from app.rag.splitter import get_text_splitter
from app.rag.vectorstore import get_vectorstore
from langchain_core.documents import Document

CHUNK_STORE_PATH = Path("data/chunks.jsonl")
CHUNK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    with CHUNK_STORE_PATH.open("a", encoding="utf-8") as f:
        for chunk in chunks:
            if "chunk_id" not in chunk.metadata:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())

            record = {
                "chunk_id": chunk.metadata["chunk_id"],
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_chunks_from_store() -> list[Document]:
    docs = []
    if not CHUNK_STORE_PATH.exists():
        return docs

    with CHUNK_STORE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                Document(
                    page_content=row["text"],
                    metadata=row["metadata"]
                )
            )
    return docs