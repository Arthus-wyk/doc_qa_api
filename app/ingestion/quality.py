from __future__ import annotations

import re

from app.ingestion.models import ChunkedUnit


def is_low_quality_chunk(unit: ChunkedUnit) -> bool:
    # 通过简单启发式规则过滤通常对检索价值较低的 chunk。
    text = unit.text.strip().lower()

    # 空文本或过短文本会增加噪声并浪费向量化 token。
    if not text:
        return True
    if len(text) < 40:
        return True
    # 过滤疑似目录（TOC）残留内容。
    if "table of content" in text or "contents" in text:
        return True
    if text.count("....") >= 3:
        return True
    if re.search(r"\.{3,}\s*\d+\s*$", text):
        return True

    return False


def filter_chunks(units: list[ChunkedUnit]) -> list[ChunkedUnit]:
    # 仅保留通过质量检查的 chunk。
    return [u for u in units if not is_low_quality_chunk(u)]

