import re
from copy import copy
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import MessagesState
from llama_index.core.schema import NodeWithScore, TextNode

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
    debug: bool
    source_file: str | None
    retrieved_docs: list[NodeWithScore]
    needs_clarification: bool
    clarification_message: str
    answer: str
    sources: list[dict[str, Any]]
    debug_info: dict[str, Any] | None


def _needs_source_clarification(question: str, source_file: str | None) -> bool:
    if source_file:
        return False
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in RELATIVE_DOC_PATTERNS)


def _node_text(item: NodeWithScore) -> str:
    node = item.node
    return getattr(node, "text", None) or node.get_content()


def _node_metadata(item: NodeWithScore) -> dict[str, Any]:
    return dict(item.node.metadata or {})


def _debug_doc_item(item: NodeWithScore) -> dict[str, Any]:
    metadata = _node_metadata(item)
    return {
        "page_content": _node_text(item)[:300],
        "metadata": metadata,
        "score": None if item.score is None else float(item.score),
    }


def _langchain_doc_to_node_with_score(doc: Any) -> NodeWithScore:
    metadata = dict(getattr(doc, "metadata", {}) or {})
    text = getattr(doc, "page_content", "")
    return NodeWithScore(node=TextNode(text=text, metadata=metadata), score=None)


def _merge_documents(primary: list[NodeWithScore], secondary: list[NodeWithScore]) -> list[NodeWithScore]:
    merged: list[NodeWithScore] = []
    seen: set[str] = set()

    for doc in [*primary, *secondary]:
        metadata = _node_metadata(doc)
        key = (
            str(metadata.get("chunk_id"))
            if metadata.get("chunk_id")
            else f"{metadata.get('source_file', '')}:{hash(_node_text(doc))}"
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
    debug_enabled = bool(state.get("debug", False))

    if _needs_source_clarification(question, source_file):
        debug_info = None
        if debug_enabled:
            debug_info = {
                "question": question,
                "rewritten_question": None,
                "k": k,
                "source_file": source_file,
                "needs_clarification": True,
                "clarification_message": CLARIFICATION_MESSAGE,
                "vector_candidates": [],
                "vector_leaf_candidates": [],
                "bm25_candidates": [],
                "merged_candidates": [],
                "reranked_candidates": [],
            }
        return {
            "retrieved_docs": [],
            "needs_clarification": True,
            "clarification_message": CLARIFICATION_MESSAGE,
            "debug_info": debug_info,
        }

    standalone_question = rewrite_query(question, llm=llm)

    vector_leaf_docs = []
    if debug_enabled:
        vector_leaf_docs = retrieve_documents(
            standalone_question,
            k=k,
            source_file=source_file,
            enable_auto_merge=False,
        )

    vector_docs = retrieve_documents(
        standalone_question,
        k=k,
        source_file=source_file,
    )

    if bm25_retriever is None:
        docs = vector_docs
        bm25_docs = []
    else:
        bm25_local = copy(bm25_retriever)
        bm25_local.k = k
        bm25_docs = [_langchain_doc_to_node_with_score(doc) for doc in bm25_local.invoke(standalone_question)]
        if source_file:
            bm25_docs = [doc for doc in bm25_docs if _node_metadata(doc).get("source_file") == source_file]
        docs = _merge_documents(bm25_docs, vector_docs)

    if source_file:
        docs = [doc for doc in docs if _node_metadata(doc).get("source_file") == source_file]

    merged_docs = list(docs)
    if docs:
        docs = rerank_documents(question, docs, top_n=10)

    debug_info = None
    if debug_enabled:
        debug_info = {
            "question": question,
            "rewritten_question": standalone_question,
            "k": k,
            "source_file": source_file,
            "needs_clarification": False,
            "vector_candidates": [_debug_doc_item(doc) for doc in vector_docs],
            "vector_leaf_candidates": [_debug_doc_item(doc) for doc in vector_leaf_docs],
            "bm25_candidates": [_debug_doc_item(doc) for doc in bm25_docs],
            "merged_candidates": [_debug_doc_item(doc) for doc in merged_docs],
            "reranked_candidates": [_debug_doc_item(doc) for doc in docs],
        }

    return {
        "retrieved_docs": docs,
        "needs_clarification": False,
        "clarification_message": "",
        "debug_info": debug_info,
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

    context = "\n\n".join(_node_text(doc) for doc in docs)

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
    debug_info = state.get("debug_info")

    sources = []
    for doc in docs:
        metadata = _node_metadata(doc)
        if doc.score is not None:
            metadata["llama_score"] = float(doc.score)
        metadata["llama_node_id"] = getattr(doc.node, "node_id", None)
        metadata["llama_is_merged"] = bool(getattr(doc.node, "child_nodes", None))
        sources.append(
            {
                "page_content": _node_text(doc)[:300],
                "metadata": metadata,
            }
        )

    result = {"sources": sources}
    if debug_info is not None:
        result["debug_info"] = debug_info
    return result
