from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingestion.models import ParsedDocument, ChunkedUnit


class BaseChunker(ABC):
    # 用于元数据与监控的可读策略名称。
    name: str = "base"

    @abstractmethod
    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # 当该切分器适用于当前解析结果时返回 True。
        raise NotImplementedError

    @abstractmethod
    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        # 将 ParsedDocument 元素转换为可检索的 chunk 单元。
        raise NotImplementedError

