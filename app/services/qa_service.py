from app.rag.retriever import get_hybrid_retriever
from app.rag.chain import get_llm, get_qa_prompt
from app.rag.reranker import rerank_documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
        "sources": docs
    }