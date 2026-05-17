from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable

from llama_index.core import Document as LlamaDocument

CHUNK_STORE_PATH = Path(os.getenv("DOC_QA_CHUNK_STORE_PATH", "data/chunks.jsonl"))
CHUNK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _normalize_record_text(item: object) -> str:
    return (
        getattr(item, "text", None)
        or getattr(item, "page_content", None)
        or ""
    )


def _normalize_record_metadata(item: object) -> dict:
    metadata = dict(getattr(item, "metadata", {}) or {})
    metadata.setdefault("chunk_id", str(uuid.uuid4()))
    return metadata


def append_chunks_to_store(chunks: Iterable[object]) -> None:
    with CHUNK_STORE_PATH.open("a", encoding="utf-8") as file_obj:
        for chunk in chunks:
            metadata = _normalize_record_metadata(chunk)
            record = {
                "chunk_id": metadata["chunk_id"],
                "text": _normalize_record_text(chunk),
                "metadata": metadata,
            }
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_from_store() -> list[LlamaDocument]:
    docs: list[LlamaDocument] = []
    if not CHUNK_STORE_PATH.exists():
        return docs

    with CHUNK_STORE_PATH.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            row = json.loads(line)
            docs.append(
                LlamaDocument(
                    text=row["text"],
                    metadata=row["metadata"],
                )
            )
    return docs
