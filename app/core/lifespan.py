from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.rag.chain import get_llm
from app.rag.vectorstore import get_vectorstore
from app.services.retriever_refresh_service import refresh_qa_retrievers


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = get_llm()
    vectorstore = get_vectorstore()

    app.state.llm = llm
    app.state.vectorstore = vectorstore
    refresh_qa_retrievers(app)

    yield
