from __future__ import annotations

import logging
from typing import Sequence

from llama_index.core import Document as LlamaDocument
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

from app.rag.llamaindex_store import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZES,
    get_embed_model,
    get_persist_dir,
    get_storage_context,
)

logger = logging.getLogger(__name__)


def build_hierarchical_index(
    documents: Sequence[LlamaDocument],
    *,
    chunk_sizes: Sequence[int] = DEFAULT_CHUNK_SIZES,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> VectorStoreIndex:
    Settings.embed_model = get_embed_model()
    storage_context = get_storage_context()

    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=list(chunk_sizes),
        chunk_overlap=chunk_overlap,
    )
    all_nodes = node_parser.get_nodes_from_documents(list(documents))
    leaf_nodes = get_leaf_nodes(all_nodes)

    storage_context.docstore.add_documents(all_nodes)
    index = VectorStoreIndex(
        leaf_nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model,
    )
    storage_context.persist(persist_dir=get_persist_dir())

    logger.info(
        "Built hierarchical index with %s total nodes and %s leaf nodes.",
        len(all_nodes),
        len(leaf_nodes),
    )
    return index
