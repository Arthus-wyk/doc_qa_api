from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument


class BaseParser(ABC):
    # 写入 DocumentMeta 的解析器标识，用于诊断与追踪。
    name: str = "base"

    @abstractmethod
    def supports(self, meta: DocumentMeta) -> bool:
        # 若该解析器可处理当前元数据描述的文件则返回 True。
        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        # 将原始文件解析为标准化 ParsedDocument。
        raise NotImplementedError

