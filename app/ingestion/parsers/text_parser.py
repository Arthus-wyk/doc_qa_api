from __future__ import annotations

from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument, ParsedElement
from app.ingestion.parsers.base import BaseParser


class TextParser(BaseParser):
    # 纯文本解析器：使用双换行作为段落边界。
    name = "text"

    def supports(self, meta: DocumentMeta) -> bool:
        return meta.file_ext in {".txt", ".log"}

    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        meta.parser_name = self.name
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        # 每个段落映射为一个 narrative_text 元素。
        elements = [
            ParsedElement(element_type="narrative_text", text=para.strip())
            for para in text.split("\n\n")
            if para.strip()
        ]
        return ParsedDocument(meta=meta, elements=elements)

