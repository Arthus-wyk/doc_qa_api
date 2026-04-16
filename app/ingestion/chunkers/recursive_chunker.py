from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.models import ParsedDocument, ChunkedUnit
from app.ingestion.chunkers.base import BaseChunker


class RecursiveFallbackChunker(BaseChunker):
    # 通用兜底切分器：当专用切分器不适用时处理任意文档。
    name = "recursive"

    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # 始终支持；作为最后一道安全网。
        return True

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        # LangChain 递归切分器会优先使用更大的分隔符，再逐步细化。
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
        )

        # 将非 TOC 文本拼成连续文本流，再切成带重叠的 chunk。
        all_text = "\n".join(
            e.text for e in parsed_doc.elements
            if e.text and not e.metadata.get("is_toc_like")
        )

        chunks = splitter.split_text(all_text)

        # 附加最小追踪元数据。
        return [
            ChunkedUnit(
                text=chunk,
                metadata={
                    "source_file": parsed_doc.meta.source_file,
                    "chunk_strategy": self.name,
                },
            )
            for chunk in chunks
            if chunk.strip()
        ]

