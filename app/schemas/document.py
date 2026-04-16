from pydantic import BaseModel

class UploadResponse(BaseModel):
    filename: str
    saved_path: str
    message: str

class IndexResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str

class Document(BaseModel):
    name:str
    ext:str
    size:int