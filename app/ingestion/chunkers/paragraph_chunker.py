from __future__ import annotations

from app.ingestion.models import ParsedDocument, ChunkedUnit
from app.ingestion.chunkers.base import BaseChunker


class ParagraphChunker(BaseChunker):
    # 按解析后的标题/段落元素对 Markdown 或文本进行切分。
    name = "paragraph"

    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # 更适合缺少稳定页码/章节结构的纯文本类格式。
        return parsed_doc.meta.file_ext in {".md", ".txt"}

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        units: list[ChunkedUnit] = []

        for element in parsed_doc.elements:
            # 保留可读内容单元，忽略空文本与非文本元素。
            if element.element_type in {"title", "narrative_text"} and element.text.strip():
                units.append(
                    ChunkedUnit(
                        text=element.text.strip(),
                        metadata={
                            "source_file": parsed_doc.meta.source_file,
                            "chunk_strategy": self.name,
                        },
                    )
                )

        return units

