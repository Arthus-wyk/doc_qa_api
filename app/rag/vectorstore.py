from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIR = "data/chroma"
COLLECTION_NAME = "documents"
logger = logging.getLogger(__name__)

def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )

def _create_vectorstore(persist_dir: str) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir
    )

def get_vectorstore():
    persist_path = Path(PERSIST_DIR)
    persist_path.mkdir(parents=True, exist_ok=True)

    try:
        return _create_vectorstore(str(persist_path))
    except Exception as first_error:
        # 常见崩溃后会残留 sqlite journal，先清理再重试一次。
        journal_path = persist_path / "chroma.sqlite3-journal"
        if journal_path.exists():
            try:
                journal_path.unlink()
                logger.warning(
                    "Detected stale Chroma journal file. Removed and retrying init.",
                    exc_info=first_error,
                )
                return _create_vectorstore(str(persist_path))
            except Exception:
                pass

        # 若仍失败，切换到新的备用目录，避免 /upload-and-ingest 整体失败。
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
            return _create_vectorstore(str(fallback_path))
        except Exception:
            logger.exception(
                "Failed to recover Chroma from %s with fallback dir %s",
                persist_path,
                fallback_path,
            )
            raise
