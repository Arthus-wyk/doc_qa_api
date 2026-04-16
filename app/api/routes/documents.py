import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Request

from app.core.exceptions import AppException
from app.core.responses import success_response
from app.schemas.document import Document
from app.utils.file_manager import RAW_DATA_DIR, save_upload_file
from app.services.retriever_refresh_service import refresh_qa_retrievers
from app.schemas.ingest import IngestResponse
from app.ingestion.service import IngestionService
import shutil

router = APIRouter()
SUPPORTED_EXTENSIONS = [
    ".pdf",
    ".txt",
    ".md",
    ".csv",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".html",
    ".htm",
    ".eml",
    ".msg",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
]
service = IngestionService()

@router.post("/upload")
async def uploadDocument(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise AppException(
            message="No file name!",
            code="INVALID_FILENAME",
            status_code=400
        )
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AppException(
            message=f"Unsupported file extension: {ext}",
            code="INVALID_FILENAME",
            status_code=400
        )

    save_path = RAW_DATA_DIR / file.filename
    save_upload_file(file, save_path)
    result = service.ingest(save_path)
    refresh_qa_retrievers(request.app)
    return success_response(
        data={
            'filename': file.filename,
            'saved_path': str(save_path),
            'chunks_indexed': result["chunks_count"],
            'parser': result["parser"],
        },
        message='File uploaded successfully'
    )


@router.post("/index")
def index_uploaded_document(filename: str, request: Request):
    file_path = RAW_DATA_DIR / filename
    if not file_path.exists():
        return success_response(
            data={'filename': None,
                  'chunks_indexed': 0
                  },
            message="Document not found"
        )

    result = service.ingest(file_path)
    refresh_qa_retrievers(request.app)

    return success_response(
        data={'filename': filename,
              'chunks_indexed': result["chunks_count"],
              'parser': result["parser"],
              },
        message="Document indexed successfully"
    )

@router.post("/upload-and-ingest")
async def upload_and_ingest(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        raise AppException(status_code=400, message="Missing filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AppException(
            message=f"Unsupported file extension: {ext}",
            code="INVALID_FILENAME",
            status_code=400
        )

    save_path = RAW_DATA_DIR / file.filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = service.ingest(save_path)
    refresh_qa_retrievers(request.app)
    return success_response(
        data=IngestResponse(**result),
        message="Document indexed successfully"
    )

@router.get("/")
def get_documents():
    data = []
    for root, dirs, files in os.walk(RAW_DATA_DIR):
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in SUPPORTED_EXTENSIONS:
                size = os.path.getsize(os.path.join(root, file))
                data.append(Document(name=name, ext=ext, size=size))

    return success_response(
        data=data,
    )



@router.get("/{filename}")
def get_documents_byName(filename: str):
    file_path = RAW_DATA_DIR / filename
    name, ext = os.path.splitext(file_path)
    if ext in SUPPORTED_EXTENSIONS:
        size = os.path.getsize(file_path)
        return success_response(
            data={'name': name, 'ext': ext, 'size': size},
        )
    else:
        raise AppException(status_code=400, message=f"File {filename} has an invalid extension.",code='INVALID_FILENAME')


@router.post("delete/{filename}")
def delete_documents_byName(filename: str):
    file_path = RAW_DATA_DIR / filename
    # 检查文件是否存在
    if not file_path.exists():
        raise AppException(status_code=500, message="File not found")

    try:
        # 尝试删除文件
        file_path.unlink()
        return success_response(
            data=None,
            message= f"File {filename} has been deleted."
        )

    except Exception as e:
        raise AppException(status_code=500, message=str(e))
