from copy import copy

from llama_index.core.schema import NodeWithScore, TextNode

from app.ingestion.chunk_store import load_chunks_from_store
from app.rag.bm25 import get_bm25_retriever
from app.rag.llamaindex_store import retrieve_documents


def _to_node_with_score(doc) -> NodeWithScore:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    text = getattr(doc, "page_content", "") or getattr(doc, "text", "")
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=None)


class LlamaIndexRetrieverAdapter:
    def __init__(self, *, k: int = 4, source_file: str | None = None):
        self.k = k
        self.source_file = source_file

    def invoke(self, query: str):
        return retrieve_documents(
            query=query,
            k=self.k,
            source_file=self.source_file,
        )


def get_retriever(k: int = 4, *, source_file: str | None = None):
    return LlamaIndexRetrieverAdapter(k=k, source_file=source_file)


def get_hybrid_retriever(k=4, *, bm25_retriever=None, source_file: str | None = None):
    vector_retriever = get_retriever(k=k, source_file=source_file)

    if bm25_retriever is None:
        docs = load_chunks_from_store()
        bm25_retriever = get_bm25_retriever(docs, k=k)
    else:
        bm25_local = copy(bm25_retriever)
        bm25_local.k = k
        bm25_retriever = bm25_local

    if bm25_retriever is None:
        return vector_retriever

    class HybridRetrieverAdapter:
        def __init__(self, *, bm25, vector, source_file_filter: str | None):
            self.bm25 = bm25
            self.vector = vector
            self.source_file_filter = source_file_filter

        def invoke(self, query: str):
            bm25_docs = [_to_node_with_score(doc) for doc in self.bm25.invoke(query)]
            vector_docs = self.vector.invoke(query)
            docs = [*bm25_docs, *vector_docs]
            if self.source_file_filter:
                docs = [doc for doc in docs if (doc.node.metadata or {}).get("source_file") == self.source_file_filter]
            unique_docs = []
            seen = set()
            for doc in docs:
                metadata = dict(doc.node.metadata or {})
                text = getattr(doc.node, "text", None) or doc.node.get_content()
                key = metadata.get("chunk_id") or (metadata.get("source_file"), text)
                if key in seen:
                    continue
                seen.add(key)
                unique_docs.append(doc)
            return unique_docs

    return HybridRetrieverAdapter(
        bm25=bm25_retriever,
        vector=vector_retriever,
        source_file_filter=source_file,
    )
