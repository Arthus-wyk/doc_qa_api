from fastapi import FastAPI

from app.core.graph import build_qa_graph
from app.ingestion.chunk_store import load_chunks_from_store
from app.rag.bm25 import get_bm25_retriever


def refresh_qa_retrievers(app: FastAPI):
    docs = load_chunks_from_store()
    bm25 = get_bm25_retriever(docs)

    app.state.bm25_retriever = bm25
    app.state.qa_graph = build_qa_graph(
        llm=app.state.llm,
        vectorstore=app.state.vectorstore,
        bm25_retriever=bm25,
    )
