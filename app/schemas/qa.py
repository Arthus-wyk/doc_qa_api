from typing import Any, Dict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    session_id: str = Field(..., description="Session ID")
    question: str = Field(..., min_length=1, description="User question")
    k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")
    debug: bool = Field(
        default=False,
        description="If true, return retrieval trace information for debugging.",
    )
    source_file: str | None = Field(
        default=None,
        description="Optional: restrict retrieval to one file, for example paper.pdf",
    )


class SourceItem(BaseModel):
    page_content: str
    metadata: Dict[str, Any]


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    retrieved_count: int
    sources: list[dict[str, Any]]
    debug_info: dict[str, Any] | None = None
