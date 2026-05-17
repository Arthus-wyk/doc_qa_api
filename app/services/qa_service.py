from app.rag.retriever import get_hybrid_retriever
from app.rag.chain import get_llm, get_qa_prompt
from app.rag.reranker import rerank_documents


def _node_text(item):
    node = item.node
    return getattr(node, "text", None) or node.get_content()


def format_docs(docs):
    return "\n\n".join(_node_text(doc) for doc in docs)

def ask_question(question: str, k: int = 4):
    retriever = get_hybrid_retriever(k=k)
    docs = retriever.invoke(question)
    docs = rerank_documents(question, docs, top_n=4)
    context = format_docs(docs)
    prompt = get_qa_prompt()
    llm = get_llm()

    messages = prompt.format_messages(
        question=question,
        context=context
    )

    response = llm.invoke(messages)

    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "page_content": _node_text(doc)[:300],
                "metadata": dict(doc.node.metadata or {}),
            }
            for doc in docs
        ],
    }
