from typing import Any, Dict

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., min_length=1, description="用户问题")
    k: int = Field(default=4, ge=1, le=10, description="检索返回的文档数")
    source_file: str | None = Field(
        default=None,
        description="可选：指定只在该文档内检索（例如 paper.pdf）",
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
