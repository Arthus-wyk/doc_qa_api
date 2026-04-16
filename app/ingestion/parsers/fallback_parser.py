from __future__ import annotations

from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument, ParsedElement
from app.ingestion.parsers.base import BaseParser


class FallbackParser(BaseParser):
    # 最后兜底解析器，避免不支持或异常文件导致流程硬失败。
    name = "fallback"

    def supports(self, meta: DocumentMeta) -> bool:
        # 恒为 True，保证路由器总有终点可选。
        return True

    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        meta.parser_name = self.name

        try:
            # 尽力按文本解码；二进制文件仍可能得到质量较差的文本。
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            # 即使文件无法按文本读取，也保持管线继续执行。
            text = ""

        # 限制文本长度，避免兜底场景下内存和 token 开销过大。
        elements = [
            ParsedElement(element_type="unknown", text=text[:50000] if text else "")
        ]

        return ParsedDocument(meta=meta, elements=elements)

