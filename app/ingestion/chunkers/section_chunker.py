from __future__ import annotations

import re
from typing import Any

from app.ingestion.chunkers.base import BaseChunker
from app.ingestion.models import ChunkedUnit, ParsedDocument


SECTION_START_RE = re.compile(r"^\s*(\d+(\.\d+)*)(?:[.)])?\s+(.+)$")
ROMAN_SECTION_RE = re.compile(r"^\s*[IVXLCM]+[.)]\s+(.+)$", re.IGNORECASE)
HEADING_SUFFIX_RE = re.compile(r"[.!?。！？:：]$")
ACADEMIC_SECTION_KEYWORDS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "method",
    "methods",
    "methodology",
    "experiment",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgments",
    "acknowledgements",
}


class SectionChunker(BaseChunker):
    # Chunk by section headings for structured PDF documents.
    name = "section"

    @staticmethod
    def _is_heading_like(text: str) -> bool:
        s = (text or "").strip()
        if not s or len(s) > 120:
            return False

        normalized = re.sub(r"\s+", " ", s).strip().lower()
        normalized_no_trailing = normalized.rstrip(".:")

        if normalized_no_trailing in ACADEMIC_SECTION_KEYWORDS:
            return True

        if SECTION_START_RE.match(s) or ROMAN_SECTION_RE.match(s):
            # Avoid sentence-like lines being treated as headings.
            if not HEADING_SUFFIX_RE.search(s):
                return True

        # Common heading style in papers, e.g. "INTRODUCTION".
        if s.isupper() and len(s) >= 4 and len(s.split()) <= 8:
            return True

        return False

    def supports(self, parsed_doc: ParsedDocument) -> bool:
        # Use multi-signal detection because parser title tags are often incomplete.
        if parsed_doc.meta.file_ext != ".pdf":
            return False

        if not parsed_doc.elements:
            return False

        sample = parsed_doc.elements[:600]
        title_count = sum(1 for e in sample if e.element_type == "title")
        heading_like_count = sum(1 for e in sample if self._is_heading_like(e.text))

        # Relaxed thresholds for real-world paper PDFs.
        return (
            title_count >= 2
            or heading_like_count >= 4
            or (title_count + heading_like_count) >= 4
        )

    def chunk(self, parsed_doc: ParsedDocument) -> list[ChunkedUnit]:
        units: list[ChunkedUnit] = []

        # 可调参数
        max_chars = getattr(self, "max_chars", 1800)
        overlap_chars = getattr(self, "overlap_chars", 250)

        # 更宽松地接纳可用于检索的文本类型
        allowed_text_types = {
            "narrative_text",
            "text",
            "list_item",
            "table_text",
            "caption",
            "uncategorized_text",
        }

        current_title: str | None = None
        current_section: str | None = None
        section_elements: list[dict[str, Any]] = []

        def _normalize_text(text: str) -> str:
            text = (text or "").strip()
            text = re.sub(r"\s+\n", "\n", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]{2,}", " ", text)
            return text.strip()

        def _safe_title(text: str | None) -> str:
            text = _normalize_text(text or "")
            return text if text else "UNTITLED_SECTION"

        def _emit_chunk(
            title: str,
            section: str | None,
            chunk_text: str,
            start_page: int | None,
            end_page: int | None,
            chunk_index: int,
        ) -> None:
            chunk_text = _normalize_text(chunk_text)
            if not chunk_text:
                return

            # 标题保留在正文里，但格式更明确，减少纯标题污染
            full_text = f"Section: {title}\n\n{chunk_text}"

            units.append(
                ChunkedUnit(
                    text=full_text,
                    metadata={
                        "title": title,
                        "section": section,
                        "start_page": start_page,
                        "end_page": end_page,
                        "page": start_page,  # 为兼容旧逻辑保留
                        "source_file": parsed_doc.meta.source_file,
                        "chunk_strategy": self.name,
                        "chunk_index": chunk_index,
                    },
                )
            )

        def _split_section_into_chunks(
            title: str,
            section: str | None,
            elements: list[dict[str, Any]],
        ) -> None:
            """
            先按 section 聚合，再按长度切块。
            """
            if not elements:
                return

            title = _safe_title(title)

            chunk_parts: list[str] = []
            chunk_pages: list[int] = []
            chunk_index = 0

            def current_len() -> int:
                return sum(len(x) for x in chunk_parts) + max(0, len(chunk_parts) - 1) * 2

            def flush_chunk() -> None:
                nonlocal chunk_parts, chunk_pages, chunk_index
                if not chunk_parts:
                    return

                chunk_text = "\n\n".join(chunk_parts)
                start_page = min(chunk_pages) if chunk_pages else None
                end_page = max(chunk_pages) if chunk_pages else None

                _emit_chunk(
                    title=title,
                    section=section,
                    chunk_text=chunk_text,
                    start_page=start_page,
                    end_page=end_page,
                    chunk_index=chunk_index,
                )
                chunk_index += 1

                # overlap：从当前块尾部截一段字符作为下一块前缀
                if overlap_chars > 0 and chunk_text:
                    overlap_text = chunk_text[-overlap_chars:].strip()
                    chunk_parts = [overlap_text] if overlap_text else []
                    chunk_pages = [end_page] if end_page is not None and overlap_text else []
                else:
                    chunk_parts = []
                    chunk_pages = []

            for e in elements:
                text = _normalize_text(e["text"])
                if not text:
                    continue

                page = e.get("page")

                # 如果单个元素本身就很长，先切成小段
                if len(text) > max_chars:
                    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
                    if not paragraphs:
                        paragraphs = [text]

                    for para in paragraphs:
                        if current_len() + len(para) + 2 > max_chars and chunk_parts:
                            flush_chunk()
                        chunk_parts.append(para)
                        if page is not None:
                            chunk_pages.append(page)

                        if current_len() >= max_chars:
                            flush_chunk()
                    continue

                # 正常元素拼接
                if current_len() + len(text) + 2 > max_chars and chunk_parts:
                    flush_chunk()

                chunk_parts.append(text)
                if page is not None:
                    chunk_pages.append(page)

            flush_chunk()

        def flush_section() -> None:
            nonlocal current_title, current_section, section_elements
            if not section_elements:
                return

            _split_section_into_chunks(
                title=current_title or "UNTITLED_SECTION",
                section=current_section,
                elements=section_elements,
            )
            section_elements = []

        for element in parsed_doc.elements:
            if element.metadata.get("is_toc_like"):
                continue

            text = _normalize_text(element.text or "")
            if not text:
                continue

            is_parser_title = element.element_type == "title"
            is_promoted_title = (
                element.element_type in {"narrative_text", "text"}
                and self._is_heading_like(text)
            )

            if is_parser_title or is_promoted_title:
                flush_section()
                current_title = text
                current_section = element.section_hint
                section_elements = []
                continue

            if element.element_type in allowed_text_types:
                # 没有标题时，也允许先进入默认 section
                if current_title is None:
                    current_title = "UNTITLED_SECTION"
                    current_section = element.section_hint

                section_elements.append(
                    {
                        "text": text,
                        "page": element.page,
                        "element_type": element.element_type,
                    }
                )

        flush_section()
        return units
