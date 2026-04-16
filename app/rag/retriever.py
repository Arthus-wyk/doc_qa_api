from copy import copy

from app.rag.bm25 import get_bm25_retriever
from app.rag.llamaindex_store import retrieve_documents
from app.services.ingest_service import load_chunks_from_store


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
            bm25_docs = self.bm25.invoke(query)
            vector_docs = self.vector.invoke(query)
            docs = [*bm25_docs, *vector_docs]
            if self.source_file_filter:
                docs = [doc for doc in docs if doc.metadata.get("source_file") == self.source_file_filter]
            unique_docs = []
            seen = set()
            for doc in docs:
                key = doc.metadata.get("chunk_id") or (doc.metadata.get("source_file"), doc.page_content)
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
