from __future__ import annotations

from pathlib import Path

from app.ingestion.models import DocumentMeta, ParsedDocument, ParsedElement
from app.ingestion.parsers.base import BaseParser


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".html",
    ".htm",
    ".eml",
    ".msg",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}


class PDFStructuredParser(BaseParser):
    # Use Unstructured for PDF/Office/HTML/Email/Image ingestion.
    # PDF is parsed by Unstructured first, then falls back to pypdf on parser/runtime issues.
    name = "unstructured_file"

    def supports(self, meta: DocumentMeta) -> bool:
        return meta.file_ext in SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        meta.parser_name = self.name
        try:
            return self._parse_with_unstructured(file_path, meta)
        except Exception as exc:
            # Keep ingestion resilient when optional dependencies/runtime are not ready.
            meta.extra["unstructured_error"] = str(exc)
            if meta.file_ext == ".pdf":
                return self._parse_pdf_with_fallback(file_path, meta)

            fallback_text = file_path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                meta=meta,
                elements=[ParsedElement(element_type="unknown", text=fallback_text[:50000])],
            )

    def _parse_with_unstructured(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        parts = self._partition_by_file_type(file_path, meta)
        elements: list[ParsedElement] = []
        max_page = 0

        for part in parts:
            text = (getattr(part, "text", None) or str(part) or "").strip()
            if not text:
                continue

            category = (getattr(part, "category", "") or "").lower()
            metadata_obj = getattr(part, "metadata", None)
            page = getattr(metadata_obj, "page_number", None)
            if isinstance(page, int):
                max_page = max(max_page, page)

            element_type = self._to_element_type(category)
            part_metadata = {"unstructured_category": category or "unknown"}
            partition_backend = meta.extra.get("partition_backend")
            if partition_backend:
                part_metadata["partition_backend"] = partition_backend
            if metadata_obj is not None and hasattr(metadata_obj, "to_dict"):
                part_metadata["unstructured_metadata"] = metadata_obj.to_dict()

            elements.append(
                ParsedElement(
                    element_type=element_type,
                    text=text,
                    page=page if isinstance(page, int) else None,
                    metadata=part_metadata,
                )
            )

        if max_page > 0:
            meta.total_pages = max_page
        return ParsedDocument(meta=meta, elements=elements)

    def _partition_by_file_type(self, file_path: Path, meta: DocumentMeta):
        ext = meta.file_ext
        filename = str(file_path)

        if ext == ".pdf":
            from unstructured.partition.pdf import partition_pdf

            meta.extra["partition_backend"] = "partition_pdf"
            return partition_pdf(filename=filename)

        if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}:
            from unstructured.partition.image import partition_image

            meta.extra["partition_backend"] = "partition_image"
            return partition_image(filename=filename)

        if ext in {".doc", ".docx"}:
            from unstructured.partition.docx import partition_docx

            meta.extra["partition_backend"] = "partition_docx"
            return partition_docx(filename=filename)

        if ext in {".ppt", ".pptx"}:
            from unstructured.partition.pptx import partition_pptx

            meta.extra["partition_backend"] = "partition_pptx"
            return partition_pptx(filename=filename)

        if ext in {".xls", ".xlsx"}:
            from unstructured.partition.xlsx import partition_xlsx

            meta.extra["partition_backend"] = "partition_xlsx"
            return partition_xlsx(filename=filename)

        if ext in {".html", ".htm"}:
            from unstructured.partition.html import partition_html

            meta.extra["partition_backend"] = "partition_html"
            return partition_html(filename=filename)

        if ext == ".eml":
            from unstructured.partition.email import partition_email

            meta.extra["partition_backend"] = "partition_email"
            return partition_email(filename=filename)

        if ext == ".msg":
            from unstructured.partition.msg import partition_msg

            meta.extra["partition_backend"] = "partition_msg"
            return partition_msg(filename=filename)

        from unstructured.partition.auto import partition

        meta.extra["partition_backend"] = "partition_auto"
        return partition(filename=filename)

    def _parse_pdf_with_fallback(self, file_path: Path, meta: DocumentMeta) -> ParsedDocument:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            meta.parser_name = "pdf_pypdf_fallback"
            meta.total_pages = len(reader.pages)

            elements: list[ParsedElement] = []
            for page_idx, page in enumerate(reader.pages, start=1):
                raw_text = page.extract_text() or ""
                for line in raw_text.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    elements.append(
                        ParsedElement(
                            element_type="narrative_text",
                            text=stripped,
                            page=page_idx,
                        )
                    )
            return ParsedDocument(meta=meta, elements=elements)
        except Exception as exc:
            meta.parser_name = "pdf_binary_fallback"
            meta.extra["pdf_fallback_error"] = str(exc)
            binary_preview = file_path.read_bytes()[:50000].decode("utf-8", errors="ignore")
            return ParsedDocument(
                meta=meta,
                elements=[ParsedElement(element_type="unknown", text=binary_preview)],
            )

    @staticmethod
    def _to_element_type(category: str) -> str:
        if category in {"title", "header"}:
            return "title"
        if category in {"listitem"}:
            return "list_item"
        if category in {"table"}:
            return "table"
        if category in {"code"}:
            return "code"
        if category in {"pagebreak"}:
            return "page_break"
        if category in {"narrativetext", "text", "uncategorizedtext"}:
            return "narrative_text"
        return "unknown"
