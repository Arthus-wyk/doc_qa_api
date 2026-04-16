from langchain_core.prompts import ChatPromptTemplate
from app.rag.chain import get_llm

def rewrite_query(question: str, chat_history=None, llm=None):
    llm = llm or get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the question to be more specific for document retrieval."),
        ("human", "{question}")
    ])

    messages = prompt.format_messages(question=question)

    response = llm.invoke(messages)
    return response.content
