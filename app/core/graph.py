from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.graph_notes import QAState, retrieve_docs_node, generate_answer_node, package_response_node


def build_qa_graph(*, llm, vectorstore, bm25_retriever=None):
    graph_builder = StateGraph(QAState)

    graph_builder.add_node(
        "retrieve_docs",
        lambda state: retrieve_docs_node(
            state,
            llm=llm,
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
        )
    )
    graph_builder.add_node(
        "generate_answer",
        lambda state: generate_answer_node(state, llm=llm)
    )
    graph_builder.add_node(
        "package_response",
        package_response_node
    )

    graph_builder.add_edge(START, "retrieve_docs")
    graph_builder.add_edge("retrieve_docs", "generate_answer")
    graph_builder.add_edge("generate_answer", "package_response")
    graph_builder.add_edge("package_response", END)

    checkpointer = MemorySaver()

    return graph_builder.compile(checkpointer=checkpointer)
