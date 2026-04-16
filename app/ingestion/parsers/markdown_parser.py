from __future__ import annotations

from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument, ParsedElement
from app.ingestion.parsers.base import BaseParser


class MarkdownParser(BaseParser):
    # 基于标题标记与非空行的轻量 Markdown 解析器。
    name = "markdown"

    def supports(self, meta: DocumentMeta) -> bool:
        return meta.file_ext in {".md", ".markdown"}

    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        meta.parser_name = self.name
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        elements: list[ParsedElement] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                # 跳过空行，让切分器只处理有意义文本。
                continue

            if stripped.startswith("#"):
                # Markdown 标题行提升为 title 元素。
                elements.append(
                    ParsedElement(
                        element_type="title",
                        text=stripped,
                        metadata={"markdown_heading": True},
                    )
                )
            else:
                # 普通非标题行作为 narrative_text 元素。
                elements.append(
                    ParsedElement(
                        element_type="narrative_text",
                        text=stripped,
                    )
                )

        return ParsedDocument(meta=meta, elements=elements)

