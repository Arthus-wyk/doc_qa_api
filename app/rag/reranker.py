import cohere
from dotenv import load_dotenv
import os

load_dotenv()

# 获取变量
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

def rerank_documents(query, docs, top_n=4):

    texts = [doc.page_content for doc in docs]

    response = co.rerank(
        query=query,
        documents=texts,
        top_n=top_n,
        model="rerank-v3.5"
    )

    reranked_docs = [docs[r.index] for r in response.results]

    return reranked_docs