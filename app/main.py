from fastapi import FastAPI, Request
from app.api.routes import health, documents, qa
from app.core.lifespan import lifespan
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.exceptions import AppException
from app.core.responses import error_response

app = FastAPI(
    title="Document QA API",
    version="1.0.0",
    description="End-to-end document question answering API built with FastAPI + LangChain",
    lifespan=lifespan
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(qa.router, prefix="/qa", tags=["qa"])

@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=exc.message,
            code=exc.code,
            details=exc.details
        ).model_dump()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=str(exc.detail),
            code="HTTP_ERROR"
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_response(
            message="Request validation failed",
            code="VALIDATION_ERROR",
            details={"errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=error_response(
            message="Internal server error",
            code="INTERNAL_SERVER_ERROR"
        ).model_dump()
    )