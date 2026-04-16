from pathlib import Path
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)

def load_documents(file_path: Path):
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif suffix in [".txt", ".md"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(file_path))
    elif suffix in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(str(file_path), mode="elements")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()
