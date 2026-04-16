import re
from copy import copy
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState

from app.rag.llamaindex_store import retrieve_documents
from app.rag.query_transform import rewrite_query
from app.rag.reranker import rerank_documents


RELATIVE_DOC_PATTERNS = (
    r"this (paper|article|document)",
    r"this text",
    r"that (paper|article|document)",
)

CLARIFICATION_MESSAGE = (
    "You referred to a specific document, but no target file is selected. "
    "Please provide `source_file` (for example: `paper.pdf`) in your request."
)


class QAState(MessagesState):
    session_id: str
    question: str
    k: int
    source_file: str | None
    retrieved_docs: list[Document]
    needs_clarification: bool
    clarification_message: str
    answer: str
    sources: list[dict[str, Any]]


def _needs_source_clarification(question: str, source_file: str | None) -> bool:
    if source_file:
        return False
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in RELATIVE_DOC_PATTERNS)


def _merge_documents(primary: list[Document], secondary: list[Document]) -> list[Document]:
    merged: list[Document] = []
    seen: set[str] = set()

    for doc in [*primary, *secondary]:
        key = (
            str(doc.metadata.get("chunk_id"))
            if doc.metadata.get("chunk_id")
            else f"{doc.metadata.get('source_file', '')}:{hash(doc.page_content)}"
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    return merged


def retrieve_docs_node(state: QAState, *, llm, vectorstore, bm25_retriever=None):
    question = state["question"]
    k = state.get("k", 16)
    source_file = state.get("source_file")

    if _needs_source_clarification(question, source_file):
        return {
            "retrieved_docs": [],
            "needs_clarification": True,
            "clarification_message": CLARIFICATION_MESSAGE,
        }

    standalone_question = rewrite_query(question, llm=llm)

    vector_docs = retrieve_documents(
        standalone_question,
        k=k,
        source_file=source_file,
    )

    if bm25_retriever is None:
        docs = vector_docs
    else:
        bm25_local = copy(bm25_retriever)
        bm25_local.k = k
        bm25_docs = bm25_local.invoke(standalone_question)
        if source_file:
            bm25_docs = [doc for doc in bm25_docs if doc.metadata.get("source_file") == source_file]
        docs = _merge_documents(bm25_docs, vector_docs)

    if source_file:
        docs = [doc for doc in docs if doc.metadata.get("source_file") == source_file]

    print("docs:", docs)
    if docs:
        docs = rerank_documents(question, docs, top_n=10)
    print("After rerank:", docs)

    return {
        "retrieved_docs": docs,
        "needs_clarification": False,
        "clarification_message": "",
    }


def generate_answer_node(state: QAState, *, llm):
    if state.get("needs_clarification"):
        message = AIMessage(content=state.get("clarification_message", CLARIFICATION_MESSAGE))
        return {
            "messages": [message],
            "answer": message.content,
        }

    docs = state.get("retrieved_docs", [])
    messages = state["messages"]

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a document QA assistant. "
                "Answer only based on the provided context. "
                "If the answer is not in the context, say you do not know.",
            ),
            MessagesPlaceholder("messages"),
            (
                "human",
                "Question: {question}\n\n"
                "Context:\n{context}",
            ),
        ]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "messages": messages,
            "context": context,
            "question": state["question"],
        }
    )

    return {
        "messages": [result],
        "answer": result.content,
    }


def package_response_node(state: QAState):
    docs = state.get("retrieved_docs", [])

    sources = []
    for doc in docs:
        sources.append(
            {
                "page_content": doc.page_content[:300],
                "metadata": doc.metadata,
            }
        )

    return {"sources": sources}
