from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.chroma import ChromaVectorStore

PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "documents"

logger = logging.getLogger(__name__)


def get_embed_model():
    # 复用你现在的 Ollama embedding
    lc_embed_model = OllamaEmbeddings(model="nomic-embed-text")
    return LangchainEmbedding(lc_embed_model)


def _create_chroma_components(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context


def get_storage_context() -> StorageContext:
    persist_path = Path(PERSIST_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)

    try:
        return _create_chroma_components(str(persist_path))
    except Exception as first_error:
        journal_path = persist_path / "chroma.sqlite3-journal"
        if journal_path.exists():
            try:
                journal_path.unlink()
                logger.warning(
                    "Detected stale Chroma journal file. Removed and retrying init.",
                    exc_info=first_error,
                )
                return _create_chroma_components(str(persist_path))
            except Exception:
                pass

        fallback_path = persist_path.parent / (
            f"{persist_path.name}_fallback_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            fallback_path.mkdir(parents=True, exist_ok=True)
            logger.error(
                "Chroma persist dir appears unavailable. Switching to fallback dir %s.",
                fallback_path,
                exc_info=first_error,
            )
            return _create_chroma_components(str(fallback_path))
        except Exception:
            logger.exception(
                "Failed to recover Chroma from %s with fallback dir %s",
                persist_path,
                fallback_path,
            )
            raise


def get_vector_index() -> VectorStoreIndex:
    Settings.embed_model = get_embed_model()
    storage_context = get_storage_context()
    return VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        embed_model=Settings.embed_model,
    )


def _node_with_score_to_document(item: NodeWithScore) -> Document:
    node = item.node
    metadata = dict(node.metadata or {})
    if item.score is not None:
        metadata["llama_score"] = float(item.score)
    text = getattr(node, "text", None) or node.get_content()
    return Document(
        page_content=text,
        metadata=metadata,
    )


def retrieve_documents(query: str, *, k: int = 4, source_file: str | None = None) -> list[Document]:
    retriever = get_vector_index().as_retriever(similarity_top_k=k)
    items = retriever.retrieve(query)
    docs = [_node_with_score_to_document(item) for item in items]
    if source_file:
        docs = [doc for doc in docs if doc.metadata.get("source_file") == source_file]
    return docs
