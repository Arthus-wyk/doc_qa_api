from __future__ import annotations

import csv
from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument, ParsedElement
from app.ingestion.parsers.base import BaseParser


class CSVTableParser(BaseParser):
    # 将 CSV 按行解析为 table 元素，每行一个元素。
    name = "csv_table"

    def supports(self, meta: DocumentMeta) -> bool:
        return meta.file_ext == ".csv"

    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        meta.parser_name = self.name

        elements: list[ParsedElement] = []

        with file_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader, start=1):
                # 将行序列化为键值文本，便于向量化与检索。
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                elements.append(
                    ParsedElement(
                        element_type="table",
                        text=row_text,
                        metadata={"row_index": row_idx, "row": row},
                    )
                )

        return ParsedDocument(meta=meta, elements=elements)

