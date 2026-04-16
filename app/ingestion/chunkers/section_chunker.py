from __future__ import annotations

import re

from app.ingestion.models import ParsedDocument, ChunkedUnit
from app.ingestion.chunkers.base import BaseChunker


SECTION_START_RE = re.compile(r"^\s*(\d+(\.\d+)+)\s+(.+)$")


class SectionChunker(BaseChunker):
    # 按章节标题切分结构化 PDF（语义聚合效果更好）。
    name = "section"

    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # 要求是 PDF 且标题信号足够，避免在非结构化文档上误判。
        title_count = sum(1 for e in parsed_doc.elements if e.element_type == "title")
        return parsed_doc.meta.file_ext == ".pdf" and title_count >= 3

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        units: list[ChunkedUnit] = []

        current_title = None
        current_section = None
        current_page = None
        buffer: list[str] = []

        def flush() -> None:
            nonlocal current_title, current_section, current_page, buffer
            # 每个章节输出一个 chunk：标题 + 缓存的正文行。
            if current_title and buffer:
                text = "\n".join([current_title] + buffer).strip()
                units.append(
                    ChunkedUnit(
                        text=text,
                        metadata={
                            "title": current_title,
                            "section": current_section,
                            "page": current_page,
                            "source_file": parsed_doc.meta.source_file,
                            "chunk_strategy": self.name,
                        },
                    )
                )
            buffer = []

        for element in parsed_doc.elements:
            # 跳过此前标记为疑似目录残留的行。
            if element.metadata.get("is_toc_like"):
                continue

            if element.element_type == "title":
                # 遇到新标题即开启新章节边界，先刷出上一章节。
                flush()
                current_title = element.text
                current_section = element.section_hint
                current_page = element.page
                buffer = []
                continue

            if element.element_type == "narrative_text":
                # 将正文行归入当前章节标题。
                if current_title:
                    buffer.append(element.text)

        # 遍历结束后刷出最后一个章节。
        flush()
        return units

