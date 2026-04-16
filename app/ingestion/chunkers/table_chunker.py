from __future__ import annotations

from app.ingestion.models import ParsedDocument, ChunkedUnit
from app.ingestion.chunkers.base import BaseChunker


class TableChunker(BaseChunker):
    # 每行表格保留一个 chunk（已由 CSV 解析器序列化）。
    name = "table"

    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # CSV 解析器产出 table 元素，因此使用表格专用切分器。
        return parsed_doc.meta.file_ext == ".csv"

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        units: list[ChunkedUnit] = []

        for element in parsed_doc.elements:
            if element.element_type == "table":
                # 保留行元数据（行号/原始列），便于检索展示与过滤。
                md = {
                    "source_file": parsed_doc.meta.source_file,
                    "chunk_strategy": self.name,
                    **element.metadata,
                }
                units.append(ChunkedUnit(text=element.text, metadata=md))

        return units

