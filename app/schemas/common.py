from typing import Any, Optional, Dict
from pydantic import BaseModel, Field


class ErrorInfo(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class MetaInfo(BaseModel):
    request_id: Optional[str] = None
    timestamp: Optional[str] = None


class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[ErrorInfo] = None
    meta: Optional[MetaInfo] = None