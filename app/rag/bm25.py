from langchain_community.retrievers import BM25Retriever

def get_bm25_retriever(docs, *, k: int = 5):
    if not docs:
        return None

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
