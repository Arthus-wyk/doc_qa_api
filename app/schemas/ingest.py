from pydantic import BaseModel


class IngestResponse(BaseModel):
    source_file: str
    parser: str
    chunk_strategy: str
    elements_count: int
    chunks_count: int