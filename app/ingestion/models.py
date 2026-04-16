from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ElementType = Literal[
    # 解析器产出的高层语义类型。
    "title",
    "narrative_text",
    "list_item",
    "table",
    "code",
    "page_break",
    "unknown",
]


@dataclass
class DocumentMeta:
    # 用于解析/切分路由与追踪的最小文档元数据。
    source_file: str
    file_ext: str
    mime_type: str | None = None
    file_size: int | None = None
    total_pages: int | None = None
    parser_name: str | None = None
    doc_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedElement:
    # 组装 chunk 前的原子解析单元（行/段落/表格行等）。
    element_type: ElementType
    text: str
    page: int | None = None
    section_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    # 标准化解析结果：公共元数据 + 有序语义元素列表。
    meta: DocumentMeta
    elements: list[ParsedElement]


@dataclass
class ChunkedUnit:
    # 最终写入向量库的向量化输入单元。
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

