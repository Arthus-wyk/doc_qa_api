from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from app.core.responses import success_response
from app.schemas.qa import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask")
def ask(payload: AskRequest, request: Request):
    graph = request.app.state.qa_graph
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=payload.question)],
            "session_id": payload.session_id,
            "question": payload.question,
            "k": payload.k,
            "source_file": payload.source_file,
        },
        config={
            "configurable": {
                "thread_id": payload.session_id
            }
        }
    )
    return success_response(
        data=AskResponse(
            session_id=payload.session_id,
            question=payload.question,
            answer=result["answer"],
            retrieved_count=len(result.get("retrieved_docs", [])),
            sources=result.get("sources", []),
        ),
        message=""
    )
