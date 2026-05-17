from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document as LangChainDocument

def get_bm25_retriever(docs, *, k: int = 5):
    if not docs:
        return None

    normalized_docs = []
    for doc in docs:
        if isinstance(doc, LangChainDocument):
            normalized_docs.append(doc)
            continue

        text = getattr(doc, "text", None)
        if text is None and hasattr(doc, "get_content"):
            text = doc.get_content()

        normalized_docs.append(
            LangChainDocument(
                page_content=text or "",
                metadata=dict(getattr(doc, "metadata", {}) or {}),
            )
        )

    bm25 = BM25Retriever.from_documents(normalized_docs)
    bm25.k = k
    return bm25
