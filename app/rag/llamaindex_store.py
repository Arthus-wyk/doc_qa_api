from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chromadb
from langchain_ollama import OllamaEmbeddings
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

PERSIST_DIR = os.getenv("DOC_QA_CHROMA_DIR", "data/chroma")
COLLECTION_NAME = "documents"
DEFAULT_CHUNK_SIZES = [2048, 512]
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_MERGE_RATIO = 0.5

logger = logging.getLogger(__name__)
_ACTIVE_PERSIST_DIR = PERSIST_DIR


def get_embed_model():
    lc_embed_model = OllamaEmbeddings(model="nomic-embed-text")
    return LangchainEmbedding(lc_embed_model,embed_batch_size=10)


def _has_persisted_hierarchy(persist_path: Path) -> bool:
    docstore_file = persist_path / "docstore.json"
    index_store_file = persist_path / "index_store.json"
    return docstore_file.exists() and index_store_file.exists()


def get_persist_dir() -> str:
    return _ACTIVE_PERSIST_DIR


def _sqlite_probe_writable(base_dir: Path) -> bool:
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        probe_db = base_dir / "_sqlite_probe.db"
        conn = sqlite3.connect(str(probe_db))
        conn.execute("CREATE TABLE IF NOT EXISTS t(v TEXT)")
        conn.commit()
        conn.close()
        probe_db.unlink(missing_ok=True)
        (base_dir / "_sqlite_probe.db-journal").unlink(missing_ok=True)
        return True
    except Exception:
        logger.warning("SQLite probe failed for directory: %s", base_dir, exc_info=True)
        return False


def _persist_dir_candidates() -> list[Path]:
    configured = Path(PERSIST_DIR)
    temp_fallback = Path(tempfile.gettempdir()) / "doc_qa_api" / "chroma"
    local_app_data = os.getenv("LOCALAPPDATA")
    local_fallback = (
        Path(local_app_data) / "doc_qa_api" / "chroma"
        if local_app_data
        else None
    )

    candidates: list[Path] = [configured, temp_fallback]
    if local_fallback is not None:
        candidates.append(local_fallback)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _create_chroma_components(persist_dir: str) -> StorageContext:
    persist_path = Path(persist_dir)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # First run / legacy data may only have vector data but no docstore/index_store files.
    # In that case we keep retrieval available and let auto-merge gracefully fall back.
    if not _has_persisted_hierarchy(persist_path):
        logger.info(
            "No persisted docstore/index_store found in %s. "
            "Using in-memory stores; auto-merge will be disabled until hierarchical index is rebuilt.",
            persist_path,
        )
        return StorageContext.from_defaults(vector_store=vector_store)

    return StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=persist_dir,
    )


def get_storage_context() -> StorageContext:
    global _ACTIVE_PERSIST_DIR

    for persist_path in _persist_dir_candidates():
        if not _sqlite_probe_writable(persist_path):
            continue

        try:
            _ACTIVE_PERSIST_DIR = str(persist_path)
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
                    _ACTIVE_PERSIST_DIR = str(persist_path)
                    return _create_chroma_components(str(persist_path))
                except Exception:
                    pass

            fallback_path = persist_path.parent / (
                f"{persist_path.name}_fallback_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            try:
                fallback_path.mkdir(parents=True, exist_ok=True)
                if not _sqlite_probe_writable(fallback_path):
                    continue

                logger.error(
                    "Chroma persist dir appears unavailable. Switching to fallback dir %s.",
                    fallback_path,
                    exc_info=first_error,
                )
                _ACTIVE_PERSIST_DIR = str(fallback_path)
                return _create_chroma_components(str(fallback_path))
            except Exception:
                logger.exception(
                    "Failed to recover Chroma from %s with fallback dir %s",
                    persist_path,
                    fallback_path,
                )
                continue

    raise RuntimeError(
        "No usable SQLite persist directory for Chroma. "
        "Please set DOC_QA_CHROMA_DIR to a local writable path (for example under %LOCALAPPDATA%)."
    )


def get_vector_index() -> VectorStoreIndex:
    Settings.embed_model = get_embed_model()
    storage_context = get_storage_context()
    return VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )


def _build_retriever(
    *,
    k: int,
    enable_auto_merge: bool = True,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
):
    storage_context = get_storage_context()
    index = get_vector_index()
    base_retriever = index.as_retriever(similarity_top_k=k)

    if not enable_auto_merge:
        return base_retriever

    try:
        return AutoMergingRetriever(
            base_retriever,
            storage_context,
            simple_ratio_thresh=merge_ratio,
            verbose=False,
        )
    except Exception:
        logger.warning(
            "Auto-merging retriever unavailable for current storage. "
            "Falling back to plain vector retrieval. "
            "You may need to rebuild the index via app.ingestion.index_builder.build_hierarchical_index().",
            exc_info=True,
        )
        return base_retriever


def _apply_source_file_filter(
    docs: Iterable[NodeWithScore],
    source_file: str | None,
) -> list[NodeWithScore]:
    if not source_file:
        return list(docs)

    return [
        doc
        for doc in docs
        if (doc.node.metadata or {}).get("source_file") == source_file
    ]


def retrieve_documents(
    query: str,
    *,
    k: int = 4,
    source_file: str | None = None,
    enable_auto_merge: bool = True,
    merge_ratio: float = DEFAULT_MERGE_RATIO,
) -> list[NodeWithScore]:
    retriever = _build_retriever(
        k=k,
        enable_auto_merge=enable_auto_merge,
        merge_ratio=merge_ratio,
    )
    items = retriever.retrieve(query)
    return _apply_source_file_filter(items, source_file)
