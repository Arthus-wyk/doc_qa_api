from datetime import datetime, timezone
from typing import Any, Optional, Dict
from app.schemas.common import APIResponse, ErrorInfo, MetaInfo


def success_response(
    data: Any = None,
    message: str = "Success",
    request_id: Optional[str] = None
) -> APIResponse:
    return APIResponse(
        success=True,
        message=message,
        data=data,
        error=None,
        meta=MetaInfo(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )


def error_response(
    message: str = "Error",
    code: str = "UNKNOWN_ERROR",
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> APIResponse:
    return APIResponse(
        success=False,
        message=message,
        data=None,
        error=ErrorInfo(
            code=code,
            details=details
        ),
        meta=MetaInfo(
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    )