import cohere
from dotenv import load_dotenv
import os
from llama_index.core.schema import NodeWithScore

load_dotenv()

# 获取变量
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

def _node_text(item: NodeWithScore) -> str:
    node = item.node
    return getattr(node, "text", None) or node.get_content()


def rerank_documents(query: str, docs: list[NodeWithScore], top_n: int = 4) -> list[NodeWithScore]:
    if not docs:
        return []

    texts = [_node_text(doc) for doc in docs]

    response = co.rerank(
        query=query,
        documents=texts,
        top_n=top_n,
        model="rerank-v3.5"
    )

    reranked_docs = [docs[r.index] for r in response.results]

    return reranked_docs
