from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    return ChatOllama(
        model="qwen2.5:7b",  # 例如 "qwen2.5:7b"
        base_url="http://localhost:11434",
        temperature=0,
    )

def get_qa_prompt():
    return ChatPromptTemplate.from_template("""
You are a document QA assistant.
Answer the user's question only using the provided context.
If the answer is not in the context, say you don't know.

Question:
{question}

Context:
{context}
""")